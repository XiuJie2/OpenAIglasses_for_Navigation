import asyncio
import websockets
import time

async def test_endpoint(name, uri, timeout=5):
    print(f"\n{'='*50}")
    print(f"🔍 Testing {name}: {uri}")
    print(f"{'='*50}")
    
    try:
        async with websockets.connect(uri) as ws:
            print(f"✅ Connected to {name}")
            
            # 嘗試接收消息
            start_time = time.time()
            try:
                while time.time() - start_time < timeout:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=1)
                        if isinstance(message, bytes):
                            print(f"📦 Received binary data, size: {len(message)} bytes")
                        else:
                            print(f"📝 Received text: {message[:100]}")
                    except asyncio.TimeoutError:
                        print(f"⏳ Waiting... (timeout in {timeout - (time.time() - start_time):.1f}s)")
                        
            except Exception as e:
                print(f"📡 Connection test complete: {e}")
                
    except Exception as e:
        print(f"❌ Failed to connect: {e}")

async def main():
    endpoints = [
        ("/ws_ui", "wss://yolo.ntubbirc.ggff.net/ws_ui"),
        ("/ws", "wss://yolo.ntubbirc.ggff.net/ws"),
        ("/ws_audio", "wss://yolo.ntubbirc.ggff.net/ws_audio"),
        ("/ws/camera", "wss://yolo.ntubbirc.ggff.net/ws/camera"),
        ("/ws/viewer", "wss://yolo.ntubbirc.ggff.net/ws/viewer"),
    ]
    
    print("🚀 Starting WebSocket endpoint tests...")
    print(f"🌐 Server: https://yolo.ntubbirc.ggff.net")
    
    for name, uri in endpoints:
        await test_endpoint(name, uri, timeout=3)
    
    print("\n" + "="*50)
    print("📋 測試完成")
    print("="*50)
    print("📌 注意事項:")
    print("  - /ws_audio 需要ESP32音頻設備")
    print("  - /ws/camera 需要ESP32攝像頭")
    print("  - /ws/viewer 需要先有ESP32攝像頭連接")
    print("  - /ws 需要UDP IMU數據源")

if __name__ == "__main__":
    asyncio.run(main())