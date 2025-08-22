package handlers

import (
	"github.com/sirupsen/logrus"
	"html/template"
	"net/http"
	"web_profile/pkg/env_config"
)

func ProfileHandler(logger *logrus.Logger) http.Handler {
	return http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			tmpl := template.Must(template.ParseFiles("static/index.html"))
			if err := tmpl.Execute(w, nil); err != nil {
				logger.Fatalf("error rendering template: %s\n", err)
			}
		},
	)
}

func RedirectHandler(config env_config.Config) http.Handler {
	return http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			http.Redirect(w, r, config.GetHttpsUrl()+r.RequestURI, http.StatusMovedPermanently)
		},
	)
}
