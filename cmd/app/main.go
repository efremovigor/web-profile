package main

import (
	"context"
	"fmt"
	"github.com/gin-gonic/gin"
	"os"
	"os/signal"
	"syscall"
	"web_profile/internal"
	"web_profile/internal/server"
	"web_profile/pkg/log"
	pkgServer "web_profile/pkg/server"
)

func main() {
	defer func() {
		if r := recover(); r != nil {
			_, _ = fmt.Fprint(os.Stderr, log.Error(fmt.Sprintf("critical error has occurred: %v", r), nil).Decorate())
		}
	}()

	ctx := context.Background()
	ctx, cancel := signal.NotifyContext(ctx, os.Interrupt, syscall.SIGTERM)
	defer cancel()
	if err := run(ctx); err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "%s\n", err)
		os.Exit(1)
	}
}

func run(ctx context.Context) (err error) {
	config := internal.NewConfig()

	level := log.DEBUG
	if !config.IsDev() {
		level = log.INFO
		gin.SetMode(gin.ReleaseMode)
	}

	logger := log.NewLogger(level, []log.Listener{log.NewConsoleListener()})
	httpServer := server.Server{
		Server: pkgServer.NewServer(
			logger,
			config,
		),
		TelegramConfig:    config,
		ImageSearchConfig: config,
	}

	httpServer.SetupRoutes = httpServer.Routing

	err = httpServer.Run(ctx)
	if err != nil {
		logger.Log(log.Exception(fmt.Sprintf("unsuccessful shutdown: %s", err), nil))
		return err
	}

	logger.Log(log.Info("successful shutdown", nil))
	return nil
}
