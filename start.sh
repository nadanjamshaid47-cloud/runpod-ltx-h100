#!/bin/bash
cd /app/ComfyUI
python main.py --listen 127.0.0.1 --port 8188 --reserve-vram 4 --fp8_e4m3fn-unet > /tmp/comfy.log 2>&1 &
sleep 45
python /app/handler.py
