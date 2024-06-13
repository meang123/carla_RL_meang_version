# RL CARLA 정리 결과 

**발표때 발표하지 않는 내용입니다 - 팀원분들이 한 내용과 다르게 진행하였기 때문에 최종 발표할때는 내용을 넣지 않기로 하였습니다**

# Custom carla Enviorment Data Flow Diagram Architecture and state behavior model 

[다이어그램_링크](https://lucid.app/lucidchart/bb920655-7a1a-49bc-86b6-37e6651eaaa7/edit?viewport_loc=-505%2C469%2C3413%2C1594%2C0_0&invitationId=inv_c42e1445-ce3e-4772-a68b-6ecf67ac5ac7)
-> 루시드 앱 로그인 하신후 보시면 됩니다 만약 링크 열리지 않으면 다시 보내드릴테니 알려주시면 감사하겠습니다 

## 환경에 대한 설명 

* Pretrained VAE 사용(RGB to VAE)
* town2 환경에서 진행하였고 다이어그램의 그림처럼 waypoint를 설정 하였습니다.

### Reward 고려 사항

+ distance from center(center factor)
+ angle(angle factor)
+ distance factor(covered distance)
-> 관련 내용은 다이어그램에 설명이 되어있습니


### 패널티 및 종료 조건 
+ collision
+ off track
+ Too fast(max speed 넘어갈시)
+ 일정 시간 동안 움직이지 않을때

Stablebaseline3 SAC ,PPO에서는 패널티 reward를 -10으로 설정했고 distance factor는 고려되지 않았습니다. 
3번 SAC 구현 및 개선에서 패널티 reward -200으로 설정했고 distance factor가 고려 되었습니다 

