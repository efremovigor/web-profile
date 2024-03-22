package main

import (
	"context"
	"fmt"
	"github.com/sirupsen/logrus"
	"golang.org/x/crypto/acme/autocert"
	"html/template"
	"io"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"web_profile/env"
)

func main() {
	ctx := context.Background()
	ctx, cancel := signal.NotifyContext(ctx, os.Interrupt, syscall.SIGTERM)
	defer cancel()
	if err := run(ctx, os.Stdout, os.Args); err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "%s\n", err)
		os.Exit(1)
	}
}
func redirectTLS(w http.ResponseWriter, r *http.Request) {
	http.Redirect(w, r, env.LoadConfig().GetHttpsUrl()+r.RequestURI, http.StatusMovedPermanently)
}

func run(ctx context.Context, w io.Writer, args []string) error {
	logger := logrus.New()
	config := env.LoadConfig()
	var wg sync.WaitGroup
	wg.Add(1)
	go func(logger *logrus.Logger) {
		defer wg.Done()
		if err := http.ListenAndServe(env.LoadConfig().GetHttpSocket(), http.HandlerFunc(redirectTLS)); err != nil {
			logger.Fatalf("ListenAndServe error: %v", err)
		}
	}(logger)

	httpServer := &http.Server{Handler: NewServer(logger)}
	if env.LoadConfig().IsDev() {
		httpServer.Addr = config.GetHttpsSocket()
	}
	wg.Add(1)
	go func(logger *logrus.Logger) {
		defer wg.Done()
		logger.Infof("listening on http://%s\n", httpServer.Addr)
		if config.IsDev() {
			if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
				logger.Fatalf("error listening and serving: %s\n", err)
			}
		} else {
			if err := httpServer.Serve(autocert.NewListener(config.Domain)); err != nil && err != http.ErrServerClosed {
				logger.Fatalf("error listening and serving: %s\n", err)
			}
		}
	}(logger)
	wg.Add(1)
	go func(logger *logrus.Logger) {
		defer wg.Done()
		<-ctx.Done()
		if err := httpServer.Shutdown(ctx); err != nil {
			logger.Fatalf("error shutting down http server: %s\n", err)
		}
	}(logger)
	wg.Wait()
	return nil
}

func NewServer(logger *logrus.Logger) http.Handler {
	mux := http.NewServeMux()
	addRoutes(
		mux,
		logger,
	)
	var handler http.Handler = mux
	return handler
}

func addRoutes(
	mux *http.ServeMux,
	logger *logrus.Logger,
) {
	mux.Handle("/", profileHandler(logger))
	mux.Handle("/full", profileFullHandler(logger))
	fs := http.FileServer(http.Dir("static"))
	mux.Handle("/static/", http.StripPrefix("/static/", fs))
}

func profileHandler(logger *logrus.Logger) http.Handler {
	return http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			tmpl := template.Must(template.ParseFiles("static/profile/profile.html", "static/index.html"))
			if err := tmpl.ExecuteTemplate(w, "index", nil); err != nil {
				logger.Fatalf("error rendering template: %s\n", err)
			}
		},
	)
}

func profileFullHandler(logger *logrus.Logger) http.Handler {
	return http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			tmpl := template.Must(template.ParseFiles("static/profile_full/profile_full.html", "static/index.html"))
			if err := tmpl.ExecuteTemplate(w, "index", nil); err != nil {
				logger.Fatalf("error rendering template: %s\n", err)
			}
		},
	)
}
