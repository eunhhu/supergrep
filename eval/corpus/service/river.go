package service

import "time"

type Pass struct {
	ExpiresAt time.Time
	Cancelled bool
}

func PassCanEnter(pass Pass, now time.Time) bool {
	return !pass.Cancelled && now.Before(pass.ExpiresAt)
}

func MayChange(projectRole string, action string) bool {
	if projectRole == "owner" {
		return true
	}
	return projectRole == "editor" && (action == "draft" || action == "comment")
}

func Alphabetical(names []string) []string {
	return names
}
