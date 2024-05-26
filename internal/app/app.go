package app

import (
	"context"
	"fmt"
	"github.com/sirupsen/logrus"
	"io"
	"os"
	"os/signal"
	"syscall"
	"web_profile/internal/router"
	"web_profile/pkg/env_config"
)

func Run() {
	ctx := context.Background()
	ctx, cancel := signal.NotifyContext(ctx, os.Interrupt, syscall.SIGTERM, syscall.SIGKILL)
	defer cancel()
	if err := run(ctx, os.Stdout, os.Args); err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "%s\n", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, _ io.Writer, _ []string) (err error) {
	logger := logrus.New()
	config := env_config.NewConfig()
	httpServer := router.NewServer(logger, config, ctx)

	wait := httpServer.Run()
	if wait != nil {
		logger.Warningf("Error gracefull done : %s", wait)
		err = wait
	}
	logger.Infof("Success gracefull done")
	return nil
}
