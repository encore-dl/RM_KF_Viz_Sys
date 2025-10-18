import pygame

from RM_KF_Viz_Sys.tools.Simulator3D import Simulator3D

# 初始化Pygame
pygame.init()

# 屏幕设置
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("3D Tracker Simulation (XY Projection)")
clock = pygame.time.Clock()

def main():
    simulator = Simulator3D()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif  event.key == pygame.K_SPACE:
                    simulator.camera.set_offset(pos_offs=(0, 0, 0), yaw_offs=0, pitch_offs=0)
                elif event.key == pygame.K_a:
                    simulator.camera.set_offset(yaw_offs=1.0)
                    print('roll')
                elif event.key == pygame.K_d:
                    simulator.camera.set_offset(yaw_offs=-1.0)
                    print('roll')
                elif event.key == pygame.K_w:
                    simulator.camera.set_offset(pitch_offs=0.5)
                    print('roll')
                elif event.key == pygame.K_s:
                    simulator.camera.set_offset(pitch_offs=-0.5)
                    print('roll')
                elif event.key == pygame.K_LEFT:
                    simulator.camera.set_offset(pos_offs=(-0.5, 0, 0))
                    print('move')
                elif event.key == pygame.K_RIGHT:
                    simulator.camera.set_offset(pos_offs=(0.5, 0, 0))
                    print('move')
                elif event.key == pygame.K_UP:
                    simulator.camera.set_offset(pos_offs=(0, 0.5, 0))
                    print('move')
                elif event.key == pygame.K_DOWN:
                    simulator.camera.set_offset(pos_offs=(0, -0.5, 0))
                    print('move')
                elif event.key == pygame.K_e:
                    simulator.auto_track_target = not simulator.auto_track_target
                elif event.key == pygame.K_MINUS:
                    simulator.camera.set_offset(pos_offs=(0, 0, -0.5))
                    print('move')
                elif event.key == pygame.K_EQUALS:
                    simulator.camera.set_offset(pos_offs=(0, 0, 0.5))
                    print('move')

        # 更新模拟器
        armors, targets = simulator.update()

        # 绘制
        simulator.draw(screen, armors, targets)

        pygame.display.flip()
        clock.tick(60)  # 60 FPS

    pygame.quit()


if __name__ == "__main__":
    main()