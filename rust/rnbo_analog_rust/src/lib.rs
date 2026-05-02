use std::f32::consts::PI;

/// A simple ADSR envelope generator
pub struct AdsrEnvelope {
    pub attack: f32,
    pub decay: f32,
    pub sustain: f32,
    pub release: f32,

    sample_rate: f32,
    state: EnvelopeState,
    level: f32,

    attack_inc: f32,
    decay_dec: f32,
    release_dec: f32,
}

#[derive(PartialEq)]
enum EnvelopeState {
    Idle,
    Attack,
    Decay,
    Sustain,
    Release,
}

impl AdsrEnvelope {
    pub fn new(sample_rate: f32, attack: f32, decay: f32, sustain: f32, release: f32) -> Self {
        let mut env = Self {
            attack,
            decay,
            sustain,
            release,
            sample_rate,
            state: EnvelopeState::Idle,
            level: 0.0,
            attack_inc: 0.0,
            decay_dec: 0.0,
            release_dec: 0.0,
        };
        env.update_rates();
        env
    }

    fn update_rates(&mut self) {
        self.attack_inc = 1.0 / (self.attack.max(0.001) * self.sample_rate);
        self.decay_dec = (1.0 - self.sustain) / (self.decay.max(0.001) * self.sample_rate);
        self.release_dec = self.sustain / (self.release.max(0.001) * self.sample_rate);
    }

    pub fn trigger_on(&mut self) {
        self.state = EnvelopeState::Attack;
    }

    pub fn trigger_off(&mut self) {
        self.state = EnvelopeState::Release;
        // recalculate release_dec based on current level to reach 0
        if self.release > 0.001 {
            self.release_dec = self.level / (self.release * self.sample_rate);
        } else {
            self.release_dec = self.level;
        }
    }

    pub fn step(&mut self) -> f32 {
        match self.state {
            EnvelopeState::Idle => {
                self.level = 0.0;
            }
            EnvelopeState::Attack => {
                self.level += self.attack_inc;
                if self.level >= 1.0 {
                    self.level = 1.0;
                    self.state = EnvelopeState::Decay;
                }
            }
            EnvelopeState::Decay => {
                self.level -= self.decay_dec;
                if self.level <= self.sustain {
                    self.level = self.sustain;
                    self.state = EnvelopeState::Sustain;
                }
            }
            EnvelopeState::Sustain => {
                // Stay at sustain level
                self.level = self.sustain;
            }
            EnvelopeState::Release => {
                self.level -= self.release_dec;
                if self.level <= 0.0 {
                    self.level = 0.0;
                    self.state = EnvelopeState::Idle;
                }
            }
        }
        self.level
    }
}

/// AnalogOscillator implementation in Rust
pub struct AnalogOscillator {
    pub sample_rate: f32,
    pub mode: u32,
    phase: f32,
}

impl AnalogOscillator {
    pub fn new(sample_rate: f32, mode: u32) -> Self {
        Self {
            sample_rate,
            mode,
            phase: 0.0,
        }
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
    }

    pub fn set_mode(&mut self, mode: u32) {
        self.mode = mode;
    }

    /// Process a block of frequencies, returning a list of audio samples
    pub fn process(&mut self, freq: Vec<f32>) -> Vec<f32> {
        let mut out = vec![0.0; freq.len()];
        self.process_block(&freq, &mut out);
        out
    }

    pub fn process_block(&mut self, freq: &[f32], out: &mut [f32]) {
        for (f, o) in freq.iter().zip(out.iter_mut()) {
            let phase_inc = f / self.sample_rate;

            let val = match self.mode {
                0 => {
                    // Noise
                    (rand::random::<f32>() * 2.0) - 1.0
                }
                1 => {
                    // Sine
                    (self.phase * 2.0 * PI).sin()
                }
                2 => {
                    // Saw
                    (self.phase * 2.0) - 1.0
                }
                3 => {
                    // Triangle
                    if self.phase < 0.5 {
                        (self.phase * 4.0) - 1.0
                    } else {
                        3.0 - (self.phase * 4.0)
                    }
                }
                4 => {
                    // Square
                    if self.phase < 0.5 { 1.0 } else { -1.0 }
                }
                5 => {
                    // Pulse (25% duty cycle for simplicity, can be parameter)
                    if self.phase < 0.25 { 1.0 } else { -1.0 }
                }
                _ => 0.0,
            };

            *o = val;

            self.phase += phase_inc;
            if self.phase >= 1.0 {
                self.phase -= 1.0;
            } else if self.phase < 0.0 {
                self.phase += 1.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analog_oscillator_saw() {
        let mut osc = AnalogOscillator::new(48000.0, 2); // Saw
        let freq = vec![12000.0; 4]; // 1/4 of sample rate, so phase increases by 0.25 per sample
        let mut out = vec![0.0; 4];

        osc.process_block(&freq, &mut out);

        // Initial phase is 0.0.
        // 1st sample: phase=0.0 -> out = 0.0 * 2.0 - 1.0 = -1.0, next phase = 0.25
        // 2nd sample: phase=0.25 -> out = 0.25 * 2.0 - 1.0 = -0.5, next phase = 0.5
        // 3rd sample: phase=0.5 -> out = 0.5 * 2.0 - 1.0 = 0.0, next phase = 0.75
        // 4th sample: phase=0.75 -> out = 0.75 * 2.0 - 1.0 = 0.5, next phase = 1.0 -> 0.0

        assert!((out[0] - (-1.0)).abs() < 1e-5);
        assert!((out[1] - (-0.5)).abs() < 1e-5);
        assert!((out[2] - (0.0)).abs() < 1e-5);
        assert!((out[3] - (0.5)).abs() < 1e-5);
    }
}
