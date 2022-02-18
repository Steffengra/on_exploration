
from numpy import (
    ndarray,
    log2,
    zeros,
)

class PunctureJob:
    def __init__(
            self,
    ) -> None:
        self.critical: int = 0  # 0==False, 1==True

    def set_critical(
            self,
    ) -> None:
        self.critical = 1


class PuncturingSimulation:
    def __init__(
            self,
            num_steps_per_frame: int,
            num_resource_blocks: int,
            minimum_occupation_length: int,
            probability_rb_occupation: float,
            probability_new_puncture_request: float,
            probability_critical_request: float,
            verbosity: int,
            rng,
    ) -> None:
        self.verbosity: int = verbosity
        self.rng = rng

        self.num_steps_per_frame: int = num_steps_per_frame
        self.num_resource_blocks: int = num_resource_blocks
        self.minimum_occupation_length: int = minimum_occupation_length
        self.probability_rb_occupation: float = probability_rb_occupation
        self.probability_new_puncture_request: float = probability_new_puncture_request
        self.probability_critical_request: float = probability_critical_request

        self.step_id: int = 0  # 0.. 6
        self.resource_block_occupation: dict = {rb_id: 0 for rb_id in range(1, num_resource_blocks + 1)}
        self.rb_power_gains: dict = {rb_id: 0 for rb_id in range(1, num_resource_blocks + 1)}
        self.puncture_queue: list = []

        self.stats: dict = {
            'tx started': 0,
            'tx interrupted': 0,
            'punctures prompted': 0,
            'punctures missed': 0,
            'critical punctures prompted': 0,
            'critical punctures missed': 0,
        }

        # INITIALIZE
        self._roll_new_rb_power_gain()
        self._update_new_channel_occupation()
        self._update_puncture_queue()

    def set_occupation(
            self,
            new_occupation: dict,
    ) -> None:
        # TEST VALIDITY-------------------------------------------------------------------------------------------------
        # correct length?
        if not len(new_occupation) == len(self.resource_block_occupation):
            if self.verbosity > 0:
                print('Error: Invalid dict length')
                exit()
        # valid entries?
        for entry in new_occupation.values():
            if (entry > self.num_steps_per_frame - self.step_id) or (entry < 0):
                print('Error: Invalid dict entry')
                exit()
        # valid keys?
        for entry in new_occupation.keys():
            if entry < 1 or type(entry) != int:
                print('Error: Invalid dict key')
                exit()
        # SET-----------------------------------------------------------------------------------------------------------
        for entry in new_occupation.keys():
            if new_occupation[entry] > 0 and self.resource_block_occupation[entry] == 0:
                self.stats['tx started'] += 1

        self.resource_block_occupation = new_occupation

    def set_puncture_job(
            self,
            critical: bool,
    ) -> None:
        if not self.puncture_queue:
            self.puncture_queue = [PunctureJob()]
            self.stats['punctures prompted'] += 1
        if critical and self.puncture_queue[0].critical == 0:
            self.puncture_queue[0].set_critical()
            self.stats['critical punctures prompted'] += 1

    def get_stats(
            self,
    ) -> dict:
        return self.stats

    def get_state(
            self,
    ) -> ndarray:
        state = zeros(3 + self.num_resource_blocks, dtype='float32')
        state[0] = self.step_id / (self.num_steps_per_frame-1)
        state[1] = 1.0 if self.puncture_queue else 0.0  # is puncturing prompt?
        state[2] = 1.0 if self.puncture_queue and self.puncture_queue[0].critical == 1 else 0.0  # puncturing prompt critical?
        state[3:3+self.num_resource_blocks] = list(self.resource_block_occupation.values())
        state[3:3+self.num_resource_blocks] = state[3:3+self.num_resource_blocks] / (self.num_steps_per_frame)

        return state

    def _calculate_reward(
            self,
            puncture_resource_block_id
    ) -> float:
        # puncture
        if puncture_resource_block_id > 0:
            self.puncture_queue = []
            if self.resource_block_occupation[puncture_resource_block_id] > 0:
                self.resource_block_occupation[puncture_resource_block_id] = 0
                self.stats['tx interrupted'] += 1

        # if critical prompt still present after puncture
        critical_puncture_miss = -0.0
        if self.puncture_queue and self.puncture_queue[0].critical == 1:
            self.stats['punctures missed'] += 1
            self.stats['critical punctures missed'] += 1
            self.puncture_queue.pop()
            print('critical puncture missed')
            critical_puncture_miss = -1000.0

        # if prompt still present at frame end after puncture
        puncture_miss = -0.0
        if self.step_id == (self.num_steps_per_frame-1) and self.puncture_queue:
            self.stats['punctures missed'] += 1
            self.puncture_queue.pop()
            print('puncture missed')
            puncture_miss = -5.0

        # sum capacity
        sum_capacity = 0
        for rb_id in self.resource_block_occupation:
            if self.resource_block_occupation[rb_id] > 0:
                sum_capacity += log2(1 + self.rb_power_gains[rb_id])

        return (
            + sum_capacity
            + puncture_miss
            + critical_puncture_miss
        )

    def _roll_new_rb_power_gain(
            self,
    ) -> None:
        if self.step_id == 0:
            for rb_id in self.rb_power_gains.keys():
                self.rb_power_gains[rb_id] = self.rng.rayleigh()**2

        # print('pg', self.rb_power_gains)

    def _update_new_channel_occupation(
            self,
    ) -> None:
        # if start of frame: fill in new occupations
        if self.step_id == 0:
            # for each channel
            for rb_id in self.resource_block_occupation.keys():
                # at a probability, fill in a job
                if self.rng.random() < self.probability_rb_occupation:
                    self.resource_block_occupation[rb_id] = self.rng.choice(range(self.minimum_occupation_length,
                                                                                  self.num_steps_per_frame+1))
                    self.stats['tx started'] += 1
        # else: reduce occupation duration by 1
        else:
            # for each channel
            for rb_id in self.resource_block_occupation.keys():
                # if occupied reduce duration by 1
                if self.resource_block_occupation[rb_id] > 0:
                    self.resource_block_occupation[rb_id] -= 1

        # print('oc', self.resource_block_occupation)

    def _update_puncture_queue(
            self
    ) -> None:
        # TODO: currently nothing stops this from throwing another puncture within a frame

        # if there is no puncture prompted
        if not self.puncture_queue:
            # at probability add prompt
            if self.rng.random() < self.probability_new_puncture_request:
                self.puncture_queue = [PunctureJob()]
                self.stats['punctures prompted'] += 1
                # at probability make prompt critical
                if self.rng.random() < self.probability_critical_request:
                    self.puncture_queue[0].set_critical()
                    self.stats['critical punctures prompted'] += 1

        # print('pq', self.puncture_queue)

    def step(
            self,
            puncture_resource_block_id,  # 0 = no puncturing
    ) -> float:

        # print('\n', self.step_id)
        # print('st', self.stats)
        reward = self._calculate_reward(puncture_resource_block_id=puncture_resource_block_id)
        print('re', reward)
        self.step_id = (self.step_id + 1) % 7
        self._roll_new_rb_power_gain()
        self._update_new_channel_occupation()
        self._update_puncture_queue()

        return reward
