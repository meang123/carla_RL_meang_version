from utils import *
import numpy as np
import typing



class PER_Buffer:
    def __init__(self, opt,
                 device='cuda:0'):  # opt is option argument in main file. device for gpu memory setting buffer content to cpu
        self.device = device

        self.buffer_size = opt.buffer_size
        self.buffer_length = 0
        self.batch_size = opt.batch_size

        self.buffer = np.empty(self.buffer_size, dtype=[("priority", np.float32), ("experience", Experience)])

        self.next_idx = 0 # 순환 버퍼 인덱스 포인터

    def __len__(self) -> int:
        return self.buffer_length

    def is_empty(self) -> bool:
        return self.buffer_length == 0

    def is_full(self) -> bool:
        return self.buffer_length == self.buffer_size

    def add(self, experience: Experience) -> None:

        """
        버퍼가 다 채워졌을때
        기존 낮은 우선순위 값을 배제 하는 방식 에서 순환 버퍼 형식으로 바꾸었다
        이를 통해 샘플링의 편향이 과도하게 발생하는것을 방지 한다
        이론상 우선순위 높은거 사용하는게 맞는데 다양성과 안정성 면에서 우선순위 낮은것을 완전히 배제하는것은 샘플 분포가 왜곡되는 문제
        오버피팅 문제 발생할수있고 이로 인해 불완정한 학습 될수있다
        :param experience:
        :return:
        """

        # Set prior 1 if empty
        priority = 1.0 if self.is_empty() else self.buffer[:self.buffer_length]["priority"].max()

        # 순환 버퍼 방식 업데이트
        self.buffer[self.next_idx] = (priority, experience)
        self.next_idx = (self.next_idx + 1) % self.buffer_size

        # 아직 버퍼가 가득 차지 않은 경우 길이 증가
        if self.buffer_length < self.buffer_size:
            self.buffer_length += 1

        # 기존 방식 : 우선순위 낮은 값 배제
        # if self.is_full():
        #     # 가장 작은 값과 바꾼다
        #     if priority >= self.buffer["priority"].min():
        #         idx = self.buffer["priority"].argmin()
        #         self.buffer[idx] = (priority, experience)
        #
        #     else:
        #         pass  # low prior not include in buffer
        # else:
        #     self.buffer[self.buffer_length] = (priority, experience)
        #     self.buffer_length += 1

    def sample(self, beta,alpha) -> typing.Tuple[np.array, np.array, np.array]:

        ps = self.buffer[:self.buffer_length]["priority"]
        sampling_probs = ps ** alpha
        sampling_probs /= sampling_probs.sum() # stochastic priority

        try:

            idxs = np.random.choice(np.arange(self.buffer_length), size=self.batch_size, replace=True, p=sampling_probs)
        except ValueError as e:
            print(f"An error occurred in np.random.choice: {e}")
            print(f"ps.size: {ps.size}, sampling_probs.size: {sampling_probs.size}, sum: {np.sum(sampling_probs)}")

        # Select experience & compute IS
        experiences = self.buffer["experience"][idxs]

        IS = (self.buffer_length * sampling_probs[idxs]) ** (-beta)
        normalized_weights = IS / IS.max()

        return idxs, experiences, normalized_weights

    def update_priorities(self, idxs: np.array, priorities: np.array) -> None:
        """Update the priorities associated with particular experiences."""
        self.buffer["priority"][idxs] = priorities
