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
	"fmt"
	"os"
	"os/signal"

	"github.com/bwmarrin/discordgo"
	"golang.org/x/oauth2/google"
)

const BotIntents = discordgo.IntentsAll

type Option struct {
	Token     string
	ProjectID string
	Location  string
}

func main() {
	ctx := context.Background()
	token := os.Getenv("DISCORD_TOKEN")
	cred, err := google.FindDefaultCredentials(ctx)
	if err != nil {
		logger.Error(fmt.Sprintf("failed to find Google Cloud application default credential: %v", err))
	}
	projectID := cred.ProjectID
	projectIDEnv := os.Getenv("PROJECT_ID")
	if projectID == "" && projectIDEnv != "" {
		projectID = projectIDEnv
	}
	o := &Option{
		Token:     token,
		ProjectID: projectID,
		Location:  "asia-east1",
	}
	logger.Info(fmt.Sprintf("configuration (project: %s, location: %s)", o.ProjectID, o.Location))

	tp, err := initTracer(o)
	if err != nil {
		logger.Error(fmt.Sprintf("failed to initialize OpenTelemetry: %v", err))
	}
	defer func() {
		if err := tp.Shutdown(context.Background()); err != nil {
			logger.Error(fmt.Sprintf("failed to shutdown TraceProvider: %v", err))
		}
	}()
	logger.Info("launched TraceProvider")

	if err := run(o); err != nil {
		logger.Error(fmt.Sprintf("server execution error: %s", err))
	}
}

func run(o *Option) error {
	logger.Info("starting Gemini discord bot")
	ctx := context.Background()
	logger.Info("create Gemini instance")
	gemini, err := NewGeminiBot(ctx, o)
	if err != nil {
		return err
	}
	logger.Info("starting discord connection")
	session, err := discordgo.New("Bot " + o.Token)
	if err != nil {
		return err
	}
	session.Identify.Intents = BotIntents
	session.AddHandler(gemini.MessageCreateHandler)
	if err = session.Open(); err != nil {
		return err
	}

	sigch := make(chan os.Signal, 1)
	signal.Notify(sigch, os.Interrupt)
	<-sigch

	logger.Info("closing discord connection")
	return session.Close()
}
