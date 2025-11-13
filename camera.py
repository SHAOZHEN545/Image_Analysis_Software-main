"""Standalone ThorCam capture tool."""

import wx

from thorcam_window import ThorCamWindow


def main() -> None:
    """Launch the ThorCam capture window."""
    app = wx.App(False)
    win = ThorCamWindow(None)
    win.Show()
    app.MainLoop()


if __name__ == "__main__":
    main()
