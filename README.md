# RL CARLA 정리 결과 


**발표때 발표하지 않는 내용입니다 - 팀원분들이 했던 내용과 다르게 진행하였기 때문에 최종 발표할때는 내용을 넣지 않기로 하였습니다**



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
> 거리 100마다 checkpoint를 설정하였기 때문에 (distanse covered / 100)*0.6으로 distance factor 고려 했습니다 angle factor와 center factor에 더 집중하기 위해 60% 정도로 제한했습니다




------------------

# 시도했던 내용(실험 내용들)

위의 다이어그램대로 구축한 칼라환경을 토대로 아래 내용을 시도 했습니다. 

## 1. stablebaseline3 SAC & PPO 

구축한 환경에서 stablebaseline3 적용했을때 작동하지 않아서 작동할수있도록 칼라 환경을 수정하였습니다 


### SAC 

SAC를 적용했을때 결과가 잘 나올줄 알았지만 생각보다 잘 나오지 않았습니다. 일정 거리를 직진을 해야 하는데 일정 스텝 지나면 직진은 조금하고 좌우로 이동하면서 track을 벗어나는 현상이 있었습니다. 

**Tensorborad 결과**

1e6(백만) 스텝까지 돌린 결과도 있었는데 결과가 좋지 않아서 다시 돌리기 위해 삭제를 했었는데 시간 부족으로 20만 스텝까지 돌리지 못했습니다 - 다음부터는 첨부 자료를 위해 삭제 하지 않겠습니다. 
결론부터 말하면 20만 스텝의 결과에서 큰 차이가 없었습니다. 

#### SAC Eval 


### PPO 
SAC가 잘 작동하지 않아서 환경문제인지 아닌지를 확인하기 위해 PPO를 사용했습니다 결론적으로 SAC보다 PPO가 더 잘 작동한것을 알수있었고 환경 문제는 아니라는것을 알수있었습니다. 


**Tensorborad 결과**

40만 스텝에서 성능이 좋았고 오히려 100만 스텝에서는 더 좋지 않았습니다 아래는 비교 영상입니다

#### 40만 스텝 Eval
#### 100만 스텝 Eval 


## 2. Raylib 환경 구축 
Stablebaseline3 SAC에서는 PER buffer를 지원하지 않았는데 Ray lib에서는 SAC_PER을 제공하는것을 확인했습니다

## 3. SAC 구현 및 개선 
## 4. DSAC-T (향후계획) 



