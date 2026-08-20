//go:build unix && !linux

package sandbox

import "syscall"

func localProcessAttributes() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{Setpgid: true}
}
