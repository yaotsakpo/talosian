import math, numpy as np, mujoco
from scene_restaurant import build_model
from restaurant_driver import RestaurantDriver
def sp(arm,tx,ty):
    m=build_model(["probe"]);d=mujoco.MjData(m);drv=RestaurantDriver(m,d,["probe"])
    drink="wine" if arm=="H" else "water"
    bn,k,pxy=drv._pick_can(arm,drink);spot=np.array([tx,ty])
    for _ in drv._serve_can(arm,bn,pxy,spot): mujoco.mj_step(m,d)
    for _ in range(300): mujoco.mj_step(m,d)
    bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,bn)
    up=d.xmat[bid].reshape(3,3)[:,2]
    t=math.degrees(math.acos(max(-1,min(1,float(up[2])))));dd=float(np.linalg.norm(d.xpos[bid][:2]-spot))
    return t,dd
# H at candidate left-of-center seats near the ring
for x,y in [(-0.16,0.22),(-0.14,0.24),(-0.12,0.24),(-0.10,0.24),(-0.10,0.26),(-0.09,0.25),(-0.13,0.20)]:
    t,dd=sp("H",x,y)
    print(f"H ({x:+.3f},{y:+.3f}) tilt={t:5.1f} dist={dd*100:4.1f}cm clean={t<15 and dd<=0.06}")
