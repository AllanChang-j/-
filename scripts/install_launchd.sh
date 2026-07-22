#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/Users/allanchang/Documents/股票數據儀表板"
TEMPLATE="/Users/allanchang/Downloads/010.2026收盤日報資料整理0713.xlsx"
PYTHON="$PROJECT_DIR/.venv/bin/python"
PLIST="$HOME/Library/LaunchAgents/com.allanchang.stock-dashboard-stage1.plist"

if [ ! -x "$PYTHON" ]; then
  echo "找不到 $PYTHON，請先建立 venv 並安裝 requirements.txt"
  exit 1
fi

cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.allanchang.stock-dashboard-stage1</string>
  <key>ProgramArguments</key>
  <array>
    <string>$PYTHON</string>
    <string>$PROJECT_DIR/src/stage1_close_report.py</string>
    <string>--template</string>
    <string>$TEMPLATE</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$PROJECT_DIR</string>
  <key>StartCalendarInterval</key>
  <array>
    <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>17</integer><key>Minute</key><integer>30</integer></dict>
    <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>17</integer><key>Minute</key><integer>30</integer></dict>
    <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>17</integer><key>Minute</key><integer>30</integer></dict>
    <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>17</integer><key>Minute</key><integer>30</integer></dict>
    <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>17</integer><key>Minute</key><integer>30</integer></dict>
  </array>
  <key>StandardOutPath</key>
  <string>$PROJECT_DIR/logs/stage1.out.log</string>
  <key>StandardErrorPath</key>
  <string>$PROJECT_DIR/logs/stage1.err.log</string>
</dict>
</plist>
PLIST

mkdir -p "$PROJECT_DIR/logs"
launchctl unload "$PLIST" >/dev/null 2>&1 || true
launchctl load "$PLIST"
echo "已安裝排程：週一至週五 17:30 執行"
