package router

import (
	"context"
	"errors"
	"github.com/sirupsen/logrus"
	"golang.org/x/crypto/acme/autocert"
	"golang.org/x/sync/errgroup"
	"net/http"
	"web_profile/internal/handlers"
	"web_profile/pkg/consts/web"
	"web_profile/pkg/env_config"
)

func NewServer(logger *logrus.Logger, config env_config.Config, ctx context.Context) Server {

	return Server{
		httpServer: &http.Server{Addr: config.GetHttpSocket(), Handler: handlers.RedirectHandler(config)},
		logger:     logger,
		config:     config,
		context:    ctx,
	}
}

type Server struct {
	httpServer *http.Server
	logger     *logrus.Logger
	config     env_config.Config
	context    context.Context
}

func (s Server) Run() error {
	g, gCtx := errgroup.WithContext(s.context)
	g.Go(func() error {
		defer gCtx.Done()
		s.logger.Infof("listening on %s\n", s.httpServer.Addr)
		if err := s.httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			s.logger.Warningf("error listening and serving: %v", err)
		}
		return nil
	})
	httpsServer := &http.Server{Handler: handRouting(s.logger)}
	if s.config.IsDev() {
		httpsServer.Addr = s.config.GetHttpsSocket()
	}
	g.Go(func() error {
		defer gCtx.Done()
		if s.config.IsDev() {
			s.logger.Infof("listening on %s://%s\n", web.HTTPProtocol, httpsServer.Addr)
			if err := httpsServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
				s.logger.Warningf("error listening and serving: %s\n", err)
			}
		} else {
			s.logger.Infof("listening on %s://%s\n", web.HTTPSProtocol, s.config.Env.Domain)
			if err := httpsServer.Serve(autocert.NewListener(s.config.Env.Domain)); err != nil && !errors.Is(err, http.ErrServerClosed) {
				s.logger.Warningf("error listening and serving: %s\n", err)
			}
		}
		return nil
	})
	g.Go(func() error {
		defer gCtx.Done()
		<-s.context.Done()
		if err := s.httpServer.Shutdown(s.context); err != nil {
			s.logger.Warningf("error shutting down http server: %s\n", err)
		}

		if err := httpsServer.Shutdown(s.context); err != nil {
			s.logger.Warningf("error shutting down https server: %s\n", err)
		}

		return nil
	})

	return g.Wait()
}

func handRouting(logger *logrus.Logger) http.Handler {
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
	mux.Handle("/", handlers.ProfileHandler(logger))
	fs := http.FileServer(http.Dir("static"))
	mux.Handle("/static/", http.StripPrefix("/static/", fs))
}
