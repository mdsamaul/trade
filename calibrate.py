"""
calibrate.py - Interactive Mouse Coordinate Calibration for Quotex Buttons.
"""

import sys
import time
import json
import urllib.request
import ctypes
from ctypes import wintypes

def attach_to_desktop():
    try:
        user32 = ctypes.windll.user32
        hDesk = user32.OpenInputDesktop(0, False, 0x01FF)
        if not hDesk:
            hDesk = user32.OpenDesktopW("Default", 0, False, 0x01FF)
        if hDesk:
            user32.SetThreadDesktop(hDesk)
            return True
    except Exception as e:
        pass
    return False

def get_mouse_pos():
    attach_to_desktop()
    pt = wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return int(pt.x), int(pt.y)

def countdown_and_capture(button_name: str, prep_seconds: int = 4, hold_seconds: int = 3):
    print(f"\n=======================================================")
    print(f"👉 আপনার মাউসটি Quotex-এর [{button_name}] বাটনের ওপর নিয়ে স্থির রাখুন!")
    print(f"   (প্রস্তুতির সময়: {prep_seconds} সেকেন্ড...)")
    print(f"=======================================================")
    
    for i in range(prep_seconds, 0, -1):
        x, y = get_mouse_pos()
        print(f"   ⏳ {i} সেকেন্ড বাকি... মাউস বাটনে রাখুন (বর্তমান: X={x}, Y={y})")
        time.sleep(1.0)
        
    print(f"\n   🎯 বাটনটি মাউস দিয়ে ধরে রাখুন (ক্যাপচার হচ্ছে)...")
    samples_x = []
    samples_y = []
    for _ in range(hold_seconds):
        x, y = get_mouse_pos()
        samples_x.append(x)
        samples_y.append(y)
        time.sleep(0.5)
        
    final_x = int(sum(samples_x) / len(samples_x))
    final_y = int(sum(samples_y) / len(samples_y))
    print(f"   ✅ ক্লিয়ার! [{button_name}] বাটনের পজিশন ক্যাপচারড: X = {final_x}, Y = {final_y}")
    return final_x, final_y

def main():
    print("\n" + "="*60)
    print("🎯 QUOTEX AUTO-CLICKER LIVE CALIBRATION STARTING...")
    print("="*60)
    time.sleep(1.0)
    
    # Step 1: Green Buy
    call_x, call_y = countdown_and_capture("🟢 GREEN BUY (CALL / UP)", prep_seconds=4, hold_seconds=2)
    
    # Step 2: Red Sell
    put_x, put_y = countdown_and_capture("🔴 RED SELL (PUT / DOWN)", prep_seconds=4, hold_seconds=2)
    
    print("\n" + "="*60)
    print("📊 চূড়ান্ত ক্যাপচারকৃত কোঅর্ডিনেট:")
    print(f"   🟢 Green Buy (CALL): X = {call_x}, Y = {call_y}")
    print(f"   🔴 Red Sell (PUT):   X = {put_x}, Y = {put_y}")
    print("="*60)
    
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
            print("\n🎉 সার্ভারে সফলভাবে সেভ ও আপডেট করা হয়েছে!")
            print(f"   কনফিগারেশন: {data.get('config', {})}")
    except Exception as e:
        print(f"\n⚠️ সার্ভার আপডেট এরর: {e}")

if __name__ == "__main__":
    main()
