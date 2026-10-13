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
        DESKTOP_ALL = 0x01FF
        hDesk = user32.OpenInputDesktop(0, False, DESKTOP_ALL)
        if not hDesk:
            hDesk = user32.OpenDesktopW("Default", 0, False, DESKTOP_ALL)
        if hDesk:
            user32.SetThreadDesktop(hDesk)
            return True
    except Exception:
        pass
    return False

def ensure_quotex_focused():
    try:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        attach_to_desktop()

        found_hwnd = None
        candidates = []
        def enum_cb(hwnd, lp):
            if user32.IsWindowVisible(hwnd):
                buf = ctypes.create_unicode_buffer(512)
                if user32.GetWindowTextW(hwnd, buf, 512) > 0:
                    title = buf.value.lower()
                    if "quotex" in title and "autonomous paper-trading" not in title:
                        candidates.append(hwnd)
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)
        user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
        if candidates:
            found_hwnd = candidates[0]

        if found_hwnd:
            cur_fg = user32.GetForegroundWindow()
            cur_thread = kernel32.GetCurrentThreadId()
            fg_thread = user32.GetWindowThreadProcessId(cur_fg, None) if cur_fg else 0
            target_thread = user32.GetWindowThreadProcessId(found_hwnd, None)

            if fg_thread and fg_thread != cur_thread:
                user32.AttachThreadInput(cur_thread, fg_thread, True)
            if target_thread and target_thread != cur_thread:
                user32.AttachThreadInput(cur_thread, target_thread, True)

            user32.AllowSetForegroundWindow(-1)
            if user32.IsIconic(found_hwnd):
                user32.ShowWindow(found_hwnd, 9)
            else:
                user32.ShowWindow(found_hwnd, 5)

            user32.BringWindowToTop(found_hwnd)
            user32.keybd_event(0x12, 0, 0, 0)
            user32.keybd_event(0x12, 0, 2, 0)
            user32.SetForegroundWindow(found_hwnd)

            if fg_thread and fg_thread != cur_thread:
                user32.AttachThreadInput(cur_thread, fg_thread, False)
            if target_thread and target_thread != cur_thread:
                user32.AttachThreadInput(cur_thread, target_thread, False)
            return True
    except Exception as e:
        print("Focus error:", e)
    return False

def auto_detect_buttons():
    try:
        from PIL import ImageGrab
        attach_to_desktop()
        ensure_quotex_focused()
        time.sleep(0.35)

        img = ImageGrab.grab(all_screens=True)
        arr = np.array(img)
        h, w, _ = arr.shape

        user32 = ctypes.windll.user32
        v_left = user32.GetSystemMetrics(76)
        v_top = user32.GetSystemMetrics(77)

        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

        g_mask = cv2.inRange(hsv, np.array([35, 80, 80]), np.array([85, 255, 255]))
        r_mask1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([12, 255, 255]))
        r_mask2 = cv2.inRange(hsv, np.array([165, 80, 80]), np.array([180, 255, 255]))
        r_mask = cv2.bitwise_or(r_mask1, r_mask2)

        def detect(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cands = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] >= 35 and cv2.boundingRect(c)[3] >= 18]
            if cands:
                bx, by, bw, bh = sorted(cands, key=lambda b: b[2] * b[3], reverse=True)[0]
                screen_x = v_left + bx + (bw // 2)
                screen_y = v_top + by + (bh // 2)
                return int(screen_x), int(screen_y)
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
