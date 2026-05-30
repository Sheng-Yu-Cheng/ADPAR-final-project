import pygame
from djitellopy import Tello
import time
# ==========================================
# 參數設定區
# ==========================================
WINDOW_SIZE = (400, 400)
FPS = 30  # 控制迴圈更新頻率（不建議設定太高，以免 UDP 封包擠塞）
SPEED = 50 # 無人機飛行速度 (0-100)
# ==========================================
# 無人機初始化
# ==========================================
tello = Tello()
tello.connect()
battery = tello.get_battery()
print(f"目前電量: {battery}%")
if battery < 20:
    print("電量過低，程式終止，請先充電。")
    tello.end()
    raise SystemExit
# ==========================================
# Pygame 初始化
# ==========================================
pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Tello 鍵盤控制介面")
clock = pygame.time.Clock()
# ==========================================
# 狀態變數 (Roll, Pitch, Throttle, Yaw)
# 對應: 左右, 前後, 上下, 旋轉
# ==========================================
lr, fb, ud, yv = 0, 0, 0, 0
print("=====================================")
print("操作說明:")
print("[T] 起飛  |  [L] 降落")
print("[W/S] 前進/後退  |  [A/D] 左平移/右平移")
print("[上/下] 上升/下降  |  [左/右] 左旋轉/右旋轉")
print("[ESC] 緊急關閉程式並降落")
print("=====================================")
running = True
is_flying = False
try:
    while running:
        # 1. 處理所有事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # --- 按鍵按下 (給予速度 / 觸發單次動作) ---
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_t and not is_flying:
                    tello.takeoff()
                    is_flying = True
                elif event.key == pygame.K_l and is_flying:
                    tello.land()
                    is_flying = False
                # 前後左右 (Pitch / Roll)
                elif event.key == pygame.K_w: fb = SPEED
                elif event.key == pygame.K_s: fb = -SPEED
                elif event.key == pygame.K_a: lr = -SPEED
                elif event.key == pygame.K_d: lr = SPEED
                # 上下與旋轉 (Throttle / Yaw)
                elif event.key == pygame.K_UP: ud = SPEED
                elif event.key == pygame.K_DOWN: ud = -SPEED
                elif event.key == pygame.K_LEFT: yv = -SPEED
                elif event.key == pygame.K_RIGHT: yv = SPEED
            # --- 按鍵放開 (速度歸零，煞車機制) ---
            elif event.type == pygame.KEYUP:
                # 偵測到對應按鍵放開時，將該軸的速度歸零
                if event.key in (pygame.K_w, pygame.K_s): fb = 0
                elif event.key in (pygame.K_a, pygame.K_d): lr = 0
                elif event.key in (pygame.K_UP, pygame.K_DOWN): ud = 0
                elif event.key in (pygame.K_LEFT, pygame.K_RIGHT): yv = 0
        # 2. 發送連續控制指令給無人機
        # 即使沒有按鍵動作，維持發送 0,0,0,0 可以避免 Tello 因為 15 秒沒有收到指令而自動降落
        if is_flying:
            tello.send_rc_control(lr, fb, ud, yv)
        # 3. 更新 UI 畫面 (目前為黑畫面，保留擴充空間)
        screen.fill((30, 30, 30))
        pygame.display.flip()
        # 4. 控制迴圈頻率
        clock.tick(FPS)
finally:
    print("正在執行安全降落程序...")
    tello.send_rc_control(0, 0, 0, 0) # 確保降落前停止所有水平運動
    tello.land()
    tello.end()
    pygame.quit()