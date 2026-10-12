"""
calibrate.py - Quotex Button Auto-Detection & Interactive Calibration.
"""

import sys
import time
import json
import urllib.request
import ctypes
from ctypes import wintypes
import cv2
import numpy as np

def attach_to_desktop():
    try:
        user32 = ctypes.windll.user32
        hDesk = user32.OpenInputDesktop(0, False, 0x01FF)
        if not hDesk:
            hDesk = user32.OpenDesktopW("Default", 0, False, 0x01FF)
        if hDesk:
            user32.SetThreadDesktop(hDesk)
            return True
    except Exception:
        pass
    return False

def ensure_quotex_focused():
    try:
        user32 = ctypes.windll.user32
        found_hwnd = None
        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)

        def enum_cb(hwnd, lp):
            nonlocal found_hwnd
            buf = ctypes.create_unicode_buffer(256)
            if user32.IsWindowVisible(hwnd) and user32.GetWindowTextW(hwnd, buf, 256) > 0:
                title = buf.value.lower()
                if "demo trading | quotex" in title or ("quotex" in title and "chrome" in title):
                    found_hwnd = hwnd
                    return False
            return True

        user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
        if found_hwnd:
            user32.ShowWindow(found_hwnd, 3)  # Maximize
            user32.SetForegroundWindow(found_hwnd)
            return True
    except Exception:
        pass
    return False

def auto_detect_buttons():
    try:
        from PIL import ImageGrab
        attach_to_desktop()
        ensure_quotex_focused()
        time.sleep(0.4)
        img = ImageGrab.grab()
        arr = np.array(img)
        h, w, _ = arr.shape
        right_panel = arr[:, int(w * 0.70):, :]
        hsv = cv2.cvtColor(right_panel, cv2.COLOR_RGB2HSV)

        g_mask = cv2.inRange(hsv, np.array([35, 100, 100]), np.array([85, 255, 255]))
        r_mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([12, 255, 255]))
        r_mask2 = cv2.inRange(hsv, np.array([165, 100, 100]), np.array([180, 255, 255]))
        r_mask = cv2.bitwise_or(r_mask1, r_mask2)

        def detect(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cands = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] >= 50 and cv2.boundingRect(c)[3] >= 20]
            if cands:
                bx, by, bw, bh = sorted(cands, key=lambda b: b[2] * b[3], reverse=True)[0]
                return int(w * 0.70) + bx + (bw // 2), by + (bh // 2)
            return None

        call_pos = detect(g_mask)
        put_pos = detect(r_mask)
        if call_pos and put_pos:
            return call_pos, put_pos
    except Exception as e:
        print("Auto scan failed:", e)
    return None, None

def save_to_server(call_x, call_y, put_x, put_y):
    payload = json.dumps({
        "enabled": True,
        "call_x": call_x,
        "call_y": call_y,
        "put_x": put_x,
        "put_y": put_y,
        "restore_cursor": False
    }).encode("utf-8")
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:8000/api/autoclick/config",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            print("\n[OK] Server configuration updated successfully!")
            print(f"     Config: {data.get('config', {})}")
            return True
    except Exception as e:
        print(f"\n[WARN] Could not update server directly: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("QUOTEX AUTO-DETECTION & CALIBRATION TOOL")
    print("="*60)
    print("Step 1: Attempting automatic Computer Vision detection...")
    
    call_pos, put_pos = auto_detect_buttons()
    if call_pos and put_pos:
        print("\nSUCCESS! Buttons automatically detected on screen:")
        print(f"  GREEN BUY (CALL / UP): X = {call_pos[0]}, Y = {call_pos[1]}")
        print(f"  RED SELL  (PUT / DOWN): X = {put_pos[0]}, Y = {put_pos[1]}")
        save_to_server(call_pos[0], call_pos[1], put_pos[0], put_pos[1])
        print("="*60)
        return

    print("\nCould not automatically detect buttons. Switching to manual calibration...")

if __name__ == "__main__":
    main()
