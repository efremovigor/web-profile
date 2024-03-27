package main

import (
	"context"
	"fmt"
	"github.com/sirupsen/logrus"
	"golang.org/x/crypto/acme/autocert"
	"golang.org/x/sync/errgroup"
	"html/template"
	"io"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"web_profile/env"
)

func main() {
	ctx := context.Background()
	ctx, cancel := signal.NotifyContext(ctx, os.Interrupt, syscall.SIGTERM, syscall.SIGKILL)
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
	httpServer := &http.Server{Addr: env.LoadConfig().GetHttpSocket(), Handler: http.HandlerFunc(redirectTLS)}

	g, gCtx := errgroup.WithContext(ctx)
	g.Go(func() error {
		defer gCtx.Done()
		logger.Infof("listening on %s\n", httpServer.Addr)
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Warningf("error listening and serving: %v", err)
		}
		return nil
	})
	httpsServer := &http.Server{Handler: NewServer(logger)}
	if env.LoadConfig().IsDev() {
		httpsServer.Addr = config.GetHttpsSocket()
	}
	g.Go(func() error {
		defer gCtx.Done()
		if config.IsDev() {
			logger.Infof("listening on http://%s\n", httpsServer.Addr)
			if err := httpsServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
				logger.Warningf("error listening and serving: %s\n", err)
			}
		} else {
			logger.Infof("listening on https://%s\n", config.Domain)
			if err := httpsServer.Serve(autocert.NewListener(config.Domain)); err != nil && err != http.ErrServerClosed {
				logger.Warningf("error listening and serving: %s\n", err)
			}
		}
		return nil
	})
	g.Go(func() error {
		defer gCtx.Done()
		<-ctx.Done()
		if err := httpServer.Shutdown(ctx); err != nil {
			logger.Warningf("error shutting down http server: %s\n", err)
		}

		if err := httpsServer.Shutdown(ctx); err != nil {
			logger.Warningf("error shutting down https server: %s\n", err)
		}

		return nil
	})
	err := g.Wait()
	if err != nil {
		logger.Warningf("Error gracefull done : %s", err)
	}
	logger.Infof("Success gracefull done")
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
