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
-> stablebaseline3_sac.py 파일에서 확인할수있습니다


### SAC 

SAC를 적용했을때 결과가 잘 나올줄 알았지만 생각보다 잘 나오지 않았습니다. 일정 거리를 직진을 해야 하는데 일정 스텝 지나면 직진은 조금하고 좌우로 이동하면서 track을 벗어나는 현상이 있었습니다. 

**Tensorborad 결과**
![image](https://github.com/meang123/carla_RL_meang_version/assets/70367965/e9662633-daf4-4c2a-9995-3c2f1d8c1f3a)
![image](https://github.com/meang123/carla_RL_meang_version/assets/70367965/16e164ab-680a-49ac-ac2d-95d41d16279a)

1e6(백만) 스텝까지 돌린 결과도 있었는데 결과가 좋지 않아서 다시 돌리기 위해 삭제를 했었는데 시간 부족으로 20만 스텝까지 돌리지 못했습니다 - 다음부터는 첨부 자료를 위해 삭제 하지 않겠습니다. 
결론부터 말하면 20만 스텝의 결과에서 큰 차이가 없었습니다. 

#### SAC Eval 
![model_300000_steps_eval](https://github.com/meang123/carla_RL_meang_version/assets/70367965/faff9c9c-6b36-428b-b50a-1b3c3106ac9c)
![model_400000_steps_eval (1)](https://github.com/meang123/carla_RL_meang_version/assets/70367965/bddd27a7-431f-4978-a73c-c130ac6bb6bb)


### PPO 
SAC가 잘 작동하지 않아서 환경문제인지 아닌지를 확인하기 위해 PPO를 사용했습니다 결론적으로 SAC보다 PPO가 더 잘 작동한것을 알수있었고 환경 문제는 아니라는것을 알수있었습니다. 


**Tensorborad 결과**
![image](https://github.com/meang123/carla_RL_meang_version/assets/70367965/6212c3da-9cca-48b2-873b-e3b1a5977d3d)
![image](https://github.com/meang123/carla_RL_meang_version/assets/70367965/1a23f07a-6982-40eb-beaa-509987a2d1c9)

40만 스텝에서 성능이 좋았고 오히려 100만 스텝에서는 더 좋지 않았습니다 아래는 비교 영상입니다

#### 40만 스텝 Eval
![model_400000_steps_eval](https://github.com/meang123/carla_RL_meang_version/assets/70367965/1803a74c-c968-488a-aa4a-b45dd253c4d8)

#### 100만 스텝 Eval 
![model_1000000_steps_eval](https://github.com/meang123/carla_RL_meang_version/assets/70367965/832fc6e3-4d97-4d7c-aca9-a7ba8437e4a5)


## 2. Raylib 환경 구축 
main.py에서 코드 확인할수있습니다

Stablebaseline3 SAC에서는 PER buffer를 지원하지 않았는데 Ray lib에서는 SAC_PER을 제공하는것을 확인했습니다 그래서 PER버퍼가 적용된 SAC알고리듬을 사용하기 위해 구축한 칼라 환경에서 ray lib를 적용하였는데 action을 계산하는 부분 compute_single_action 부분에서 알수없는 에러가 발생하였고 시간이 부족한 상태에서 계속 환경 맞추는 작업을 이어서 하기에는 좋지 않다고 판단했기 때문에 우선 포기를 하고 다음으로 넘어갔습니

## 3. SAC 구현 및 개선 

main_ver2.py에서 확인할수있습니다.
Ray lib 환경과 맞추는 작업을 실패하였고 SAC성능을 올리기 위해 PER 및 개선할수있는 부분을 개선해서 직접 코드 구현을 했고 환경과 맞추는 작업을 진행했습니다

### 변경된 내용

+ Replay buffer -> PER buffer
+ critic network -> RBF critic network 변경

[RBF_적용근거](https://ar5iv.labs.arxiv.org/html/2107.13356)
해당 논문에서는 continuous action space기반에 행동이 많이 있을때  value based model 과 actor critic model을 비교 하는 내용이고 RBF를 DQN에 적용한 value based model이 actor critic net (TD3, SAC)와 비교 했을때 더 좋은 결과를 보였다는 내용입니다 위에서는 RBF DQN value based vs actor critic method 방식의 비교 차이를 중점으로 작성된 논문인데 SAC에 적용하기 위해 actor policy net은 그대로 두고 critic net을 Q net -> RBF net으로 구성하여 적용하였습니다.

Gaussian RBF 적용했습니다 

![image](https://github.com/meang123/carla_RL_meang_version/assets/70367965/a01be58d-f312-4c44-964c-d01104a664c3)


a_i 는 action space 안에서 centroid location (state와 theta로부터 learn)

- Determine the weighting of each value
- Distance to action을 통해 가중치 계산에 영향준다

|a-A_i| action A와 centroid a_i의 유사도를 봅니다

v_i : state주어졌을때 values(Expected reward or value function at each entroid location)

- Expected value or reward at each centroid location
- Q value estimate를 구하기 위해 가중치 부여하고 합산하는 양

각 state마다 Action space안에서 centroid 배치 위치 학습

각 state마다 centroid value에 어떤 값 넣어야 하는 값 학습

-> 전반적으로 Q function approximation을 RBF를 통해하는게 목표입니다. 

### Tensorboard 결과 
![image](https://github.com/meang123/carla_RL_meang_version/assets/70367965/04472369-bf40-4591-903d-a1692d544a35)


### Train 시작 영상 
Stbablebaseline3 SAC를 train했을때는 앞서 언급한것과 같이 불안정한 모습을 보였는데 개선한 SAC를 작동했을때 처음 부터 안정적으로 나아가는 모습을 알수있었습니다. 



https://github.com/meang123/carla_RL_meang_version/assets/70367965/29a618ea-8e69-4026-be2c-d4221e9d2bd8



### Eval 영상 


https://github.com/meang123/carla_RL_meang_version/assets/70367965/4d04712d-c571-4713-9d6a-61f36ca6723d



-> 거의 1e6(백만)을 학습 시켰을때의 결과입니다 

+ stablebaseline3 PPO,SAC처럼 좌우로 움직이면서 움직이지는 않음
+ 직선으로 직진하지 않고 코너를 돌려는 모습이 보임

학습을 더 진행했을때 좋아질것으로 예상합니다 시간 부족으로 더 학습을 진행하지 못했는데 3백만까지 학습을 시켰을때는 코너를 돌수있을것으로 예상하고 있는데 향후에 진행하여 결과 비교 하는 영상을 업로드 하겠습니다 



## 4. DSAC-T (향후계획) 

[DAC-T](https://arxiv.org/abs/2310.05858)

앞서 언급했던것과 같이 Stablebaseline3 SAC train 했을때 일정 타임 스텝 지났을때 불안정한 현상(좌우로 이동하면서 Track 벗어남을 반복)을 발견할수있었습니다. 그래서 더 개선할수있는 방법을 탐색하다가 위에 논문을 발견했습니다. 분포를 기반으로 SAC를 적용하는 방식인 DSAC version 1을 개선한 DSAC version 2가 DSAC T입니다. 내용을 파악하고 코드를 적용하기에는 시간이 부족해서 향후 계획으로 남겼습니다. 




