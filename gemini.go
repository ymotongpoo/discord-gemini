// Copyright 2024 Yoshi Yamaguchi
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"slices"

	"cloud.google.com/go/vertexai/genai"
	"github.com/bwmarrin/discordgo"
)

const (
	// ModelName is the name of generative AI model for Gemini API.
	// The names of models are listed on this page.
	// https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models
	ModelName   = "gemini-1.5-pro-001"
	Temperature = 0.4

	// LimitConditionPrompt is the supplementary prompt to limit the size and format
	// of the response from the model not to exceed the discord chat message size.
	LimitConditionPrompt = "返答は合計2000文字以内にしてください。また出力形式はプレーンテキストにしてください。"
)

var (
	// mentionPtn is a regular expression pattern to match mention strings
	// in discord chat messages.
	mentionPtn = regexp.MustCompile(`<@[0-9]+>`)
)

// GeminiBot is a struct to hold the stateful data of Gemini API
type GeminiBot struct {
	client *genai.Client
	model  *genai.GenerativeModel
}

// NewGeminiBot creates a discord bot instance attached with a Geimin model.
// The model is enabled with function call declaration for external web page extraction.
func NewGeminiBot(ctx context.Context, o *Option) (*GeminiBot, error) {
	client, err := genai.NewClient(ctx, o.ProjectID, o.Location)
	if err != nil {
		return nil, err
	}
	model := client.GenerativeModel(ModelName)

	fetchWebsiteContent := &genai.FunctionDeclaration{
		Name:        "fetchWebsiteContent",
		Description: "プロンプト中で指定されたURLにあるページの内容を取得する関数",
		Parameters: &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"url": {
					Type:        genai.TypeString,
					Description: "ページの内容を取得したいURL",
				},
			},
		},
	}
	model.Tools = []*genai.Tool{
		{FunctionDeclarations: []*genai.FunctionDeclaration{fetchWebsiteContent}},
	}

	return &GeminiBot{
		client: client,
		model:  model,
	}, nil
}

// Chat is the wrapper to post a query to Gemini.
// Internally, it let Gemini model to call function in addition,
// and through extra context with function call response if necessary.
func (g *GeminiBot) Chat(prompt string) (string, error) {
	ctx := context.Background()
	chat := g.model.StartChat()
	resp, err := chat.SendMessage(ctx, genai.Text(prompt+LimitConditionPrompt))
	if err != nil {
		return "", fmt.Errorf("failed to call: %v", err)
	}
	parts := resp.Candidates[0].Content.Parts
	if len(resp.Candidates) == 0 || len(parts) == 0 {
		return "", errors.New("empty response from model")
	}
	frs, err := g.handleFunctionCalls(parts)
	if err != nil {
		return "", err
	}
	logger.Info(fmt.Sprintf("function response length: %v", len(frs)))
	if len(frs) == 0 {
		v, ok := parts[0].(genai.Text)
		if !ok {
			return "", err
		}
		return string(v), nil
	}
	resp2, err := chat.SendMessage(ctx, frs...)
	if err != nil {
		return "", fmt.Errorf("failed to send function call's response: %v", err)
	}
	part2 := resp2.Candidates[0].Content.Parts[0]
	ret, ok := part2.(genai.Text)
	if !ok {
		return "", fmt.Errorf("failed to second response data to Text: %v", resp2)
	}
	return string(ret), nil
}

func (g *GeminiBot) handleFunctionCalls(parts []genai.Part) ([]genai.Part, error) {
	ps := []genai.Part{}
	for _, p := range parts {
		if _, ok := p.(genai.FunctionCall); ok {
			ps = append(ps, p)
		}
	}
	frs := []genai.Part{}
	for _, p := range ps {
		fc, _ := p.(genai.FunctionCall)
		switch fc.Name {
		case "fetchWebsiteContent":
			data, err := fetchWebsiteContentFunc(fc.Args)
			if err != nil {
				return nil, err
			}
			frs = append(frs, genai.FunctionResponse{
				Name: "fetchWebisiteContent",
				Response: map[string]any{
					"content": string(data),
				},
			})
		}
	}
	return frs, nil
}

// MessageCreateHandler is the discord bot handler for message creation event
func (g *GeminiBot) MessageCreateHandler(s *discordgo.Session, m *discordgo.MessageCreate) {
	// check if this is a mention to this bot
	if !slices.ContainsFunc(m.Mentions, func(u *discordgo.User) bool {
		return s.State.User.ID == u.ID
	}) {
		return
	}
	// remove mentions from the original query from the discord
	content := m.Content
	replaced := mentionPtn.ReplaceAllString(content, "")
	resp, err := g.Chat(replaced)
	if err != nil {
		logger.Error(fmt.Sprintf("failed to call Gemini: %v", err))
	}
	s.ChannelMessageSend(m.ChannelID, resp)
}

// fetchWebsiteContentFunc accepts genai.FunctionResponse.Response and
// returns the content of the web page.
func fetchWebsiteContentFunc(args map[string]any) ([]byte, error) {
	url, ok := args["url"].(string)
	if !ok {
		return nil, fmt.Errorf("failed to convert URL data: %v", args["url"])
	}
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	content, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	return []byte(content), nil
}
