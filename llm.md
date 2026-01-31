# Complete LLM Lineage: Dependency-Ordered Checklist

Every concept appears only after all concepts it depends on. Papers are listed with their prerequisites.

---

## PART 1: MATHEMATICAL & COMPUTATIONAL FOUNDATIONS

### 1.1 Core Mathematics
- [x] 1. Linear Algebra (matrix multiplication, vectors, tensors)
- [x] 2. Calculus (derivatives, chain rule)
- [x] 3. Probability Theory (distributions, sampling, softmax)
- [x] 4. Information Theory (entropy, cross-entropy, KL divergence)
- [x] 5. Optimization Theory (convex optimization, gradients)

### 1.2 Neural Network Foundations
- [x] 6. Perceptron
- [x] 7. Activation Functions (sigmoid, tanh)
- [x] 8. Multi-layer Perceptron (MLP)
- [x] 9. Backpropagation
- [x] 10. Gradient Descent
- [x] 11. Stochastic Gradient Descent (SGD)
- [x] 12. Momentum
- [ ] 13. Adam Optimizer
- [ ] 14. AdamW (weight decay decoupled)
- [x] 15. Learning Rate
- [x] 16. Learning Rate Scheduling (warmup, decay)
- [x] 17. Batch Size
- [x] 18. Epochs
- [x] 19. Overfitting / Underfitting
- [x] 20. Regularization
- [x] 21. Dropout
- [ ] 22. Weight Decay
- [ ] 23. Gradient Clipping
- [ ] 24. Batch Normalization
- [x] 25. Loss Functions (MSE, cross-entropy)
- [ ] 26. Label Smoothing

### 1.3 Embeddings
- [x] 27. One-hot Encoding
- [x] 28. Word Embeddings (Word2Vec, GloVe concepts)
- [ ] 29. Learned Embeddings
- [ ] 30. Embedding Dimension
- [x] 31. Vocabulary
- [x] 32. Token
- [x] 33. Tokenization (basic concept)

---

## PART 2: SEQUENCE MODELING FOUNDATIONS

### 2.1 Recurrent Architectures
- [x] 34. Recurrent Neural Network (RNN) — *Requires: 8, 9, 10*
- [x] 35. Vanishing Gradient Problem — *Requires: 34, 9*
- [x] 36. Exploding Gradient Problem — *Requires: 34, 9*
- [x] 37. Hidden State — *Requires: 34*
- [x] 38. LSTM (Long Short-Term Memory) — Hochreiter & Schmidhuber, 1997 — *Requires: 34, 35, 37*
- [x] 39. Forget Gate — *Requires: 38*
- [x] 40. Input Gate — *Requires: 38*
- [x] 41. Output Gate — *Requires: 38*
- [x] 42. Cell State — *Requires: 38*
- [ ] 43. GRU (Gated Recurrent Unit) — Cho et al., 2014 — *Requires: 38*
- [ ] 44. Bidirectional RNN — *Requires: 34*

### 2.2 Sequence-to-Sequence
- [x] 45. Encoder (sequence models) — *Requires: 34 or 38*
- [x] 46. Decoder (sequence models) — *Requires: 34 or 38*
- [x] 47. Encoder-Decoder Architecture — *Requires: 45, 46*
- [ ] 48. Seq2Seq — Sutskever et al., 2014 — *Requires: 38, 47*
- [ ] 49. Context Vector (fixed-length bottleneck) — *Requires: 48*
- [ ] 50. Teacher Forcing — *Requires: 46*

### 2.3 Attention Mechanism Origins
- [x] 51. Attention (general concept) — *Requires: 47, 49*
- [ ] 52. Bahdanau Attention (Additive Attention) — Bahdanau et al., 2015 — *Requires: 48, 51*
- [ ] 53. Alignment Scores — *Requires: 52*
- [x] 54. Attention Weights — *Requires: 52, 3 (softmax)*
- [ ] 55. Context Vector (dynamic, attention-weighted) — *Requires: 52, 54*
- [ ] 56. Luong Attention (Multiplicative Attention) — Luong et al., 2015 — *Requires: 52*

---

## PART 3: THE TRANSFORMER

### 3.1 Core Transformer Components
- [x] 57. Query (Q) — *Requires: 51, 1*
- [x] 58. Key (K) — *Requires: 51, 1*
- [x] 59. Value (V) — *Requires: 51, 1*
- [x] 60. Scaled Dot-Product Attention — *Requires: 57, 58, 59, 3 (softmax)*
- [x] 61. Attention Head — *Requires: 60*
- [x] 62. Multi-Head Attention (MHA) — *Requires: 61*
- [x] 63. Projection Matrices (W_Q, W_K, W_V, W_O) — *Requires: 62, 1*
- [x] 64. Residual Connection (Skip Connection) — *Requires: 8*
- [x] 65. Layer Normalization — *Requires: 24 (concept)*
- [x] 66. Feed-Forward Network (FFN) in Transformers — *Requires: 8, 6 (activation)*
- [x] 67. ReLU Activation — *Requires: 6*
- [ ] 68. Positional Encoding (concept) — *Requires: 29*
- [ ] 69. Sinusoidal Positional Encoding — *Requires: 68*
- [ ] 70. **Transformer — Vaswani et al., "Attention Is All You Need", 2017** — *Requires: 62, 63, 64, 65, 66, 67, 69*

### 3.2 Transformer Variants
- [x] 71. Self-Attention — *Requires: 60, 70*
- [x] 72. Cross-Attention — *Requires: 60, 70*
- [x] 73. Causal Masking (Autoregressive Masking) — *Requires: 71*
- [x] 74. Decoder-Only Architecture — *Requires: 70, 73*
- [x] 75. Encoder-Only Architecture — *Requires: 70, 71*
- [x] 76. Encoder-Decoder Architecture (Transformer) — *Requires: 70, 71, 72*

### 3.3 Normalization Variants
- [x] 77. Post-Layer Normalization (Post-LN) — *Requires: 65, 70*
- [ ] 78. Pre-Layer Normalization (Pre-LN) — *Requires: 65, 70*
- [ ] 79. RMSNorm (Root Mean Square Normalization) — Zhang & Sennrich, 2019 — *Requires: 65*
- [ ] 80. DeepNorm — Wang et al., 2022 — *Requires: 65, 64*

### 3.4 Activation Functions
- [ ] 81. GELU (Gaussian Error Linear Unit) — Hendrycks & Gimpel, 2016 — *Requires: 67*
- [ ] 82. Swish Activation — *Requires: 67*
- [ ] 83. Gated Linear Unit (GLU) — *Requires: 66*
- [ ] 84. SwiGLU — Shazeer, 2020 — *Requires: 82, 83*

### 3.5 Positional Encoding Evolution
- [ ] 85. Learned Positional Embeddings — *Requires: 68, 29*
- [ ] 86. Relative Position Encoding — Shaw et al., 2018 — *Requires: 68, 60*
- [ ] 87. Transformer-XL — Dai et al., 2019 — *Requires: 70, 86*
- [ ] 88. Rotary Position Embedding (RoPE) — Su et al., 2021 — *Requires: 68, 86, 1 (rotation matrices)*
- [ ] 89. ALiBi (Attention with Linear Biases) — Press et al., 2021 — *Requires: 60, 68*
- [ ] 90. Position Interpolation — Chen et al., 2023 — *Requires: 88*
- [ ] 91. NTK-aware RoPE Scaling — 2023 — *Requires: 88*
- [ ] 92. YaRN (Yet another RoPE extensioN) — Peng et al., 2023 — *Requires: 88, 91*
- [ ] 93. LongRoPE — Ding et al., 2024 — *Requires: 88, 90*

---

## PART 4: TOKENIZATION

- [x] 94. Byte Pair Encoding (BPE) — Sennrich et al., 2016 — *Requires: 33*
- [ ] 95. WordPiece — Schuster & Nakajima, 2012 — *Requires: 33*
- [ ] 96. SentencePiece — Kudo & Richardson, 2018 — *Requires: 94*
- [ ] 97. Unigram Language Model (tokenization) — *Requires: 33*
- [ ] 98. Byte-level BPE — *Requires: 94*
- [ ] 99. Tiktoken — OpenAI — *Requires: 94*

---

## PART 5: PRE-TRAINING PARADIGMS

### 5.1 Language Modeling Objectives
- [ ] 100. Autoregressive Language Modeling (Causal LM)
- [ ] 101. Masked Language Modeling (MLM)
- [ ] 102. Span Corruption — Raffel et al. (T5), 2020
- [ ] 103. Prefix Language Modeling
- [ ] 104. Fill-in-the-Middle (FIM)
- [ ] 105. UL2 (Unified Language Learner) — Tay et al., 2022

### 5.2 Foundational Models
- [ ] 106. **GPT-1 — Radford et al., OpenAI, 2018**
- [ ] 107. **BERT — Devlin et al., Google, 2018**
- [ ] 108. **GPT-2 — Radford et al., OpenAI, 2019**
- [ ] 109. **T5 — Raffel et al., Google, 2020**
- [ ] 110. **GPT-3 — Brown et al., OpenAI, 2020**

### 5.3 Pre-training Data
- [ ] 111. Common Crawl
- [ ] 112. The Pile — EleutherAI, 2020
- [ ] 113. Perplexity Filtering
- [ ] 114. Deduplication (MinHash, exact matching)
- [ ] 115. Quality Classifier Filtering
- [ ] 116. RedPajama
- [ ] 117. SlimPajama
- [ ] 118. FineWeb — HuggingFace, 2024
- [ ] 119. DCLM (DataComp-LM) — 2024
- [ ] 120. Data Mixing Ratios
- [ ] 121. Data Annealing
- [ ] 122. Curriculum Learning

---

## PART 6: TRAINING INFRASTRUCTURE

### 6.1 Parallelism
- [ ] 123. Data Parallelism (DP)
- [ ] 124. Tensor Parallelism (TP) — Megatron-LM
- [ ] 125. Pipeline Parallelism (PP)
- [ ] 126. Micro-batching
- [ ] 127. ZeRO (Zero Redundancy Optimizer) — DeepSpeed
- [ ] 128. ZeRO Stage 1 (Optimizer State Partitioning)
- [ ] 129. ZeRO Stage 2 (+ Gradient Partitioning)
- [ ] 130. ZeRO Stage 3 (+ Parameter Partitioning)
- [ ] 131. FSDP (Fully Sharded Data Parallel) — PyTorch
- [ ] 132. Gradient Checkpointing (Activation Recomputation)

### 6.2 Precision
- [ ] 133. FP32 (Full Precision)
- [ ] 134. FP16 (Half Precision)
- [ ] 135. BF16 (Brain Float 16)
- [ ] 136. Mixed Precision Training
- [ ] 137. Loss Scaling
- [ ] 138. FP8 Training
- [ ] 139. High-Precision Accumulation

### 6.3 Training Stability
- [ ] 140. Loss Spike Detection
- [ ] 141. Checkpoint Rollbacks
- [ ] 142. NaN Handling
- [ ] 143. Gradient Explosion Prevention

### 6.4 Initialization
- [ ] 144. Xavier/Glorot Initialization
- [ ] 145. He Initialization
- [ ] 146. Small Initialization for Output Layers
- [ ] 147. Residual Scaling (GPT-2 style)

### 6.5 Advanced Optimizers
- [ ] 148. Muon Optimizer — Moonshot, 2025
- [ ] 149. MuonClip — Moonshot, 2025
- [ ] 150. μP (Maximal Update Parameterization)

---

## PART 7: ATTENTION EFFICIENCY

### 7.1 KV Cache
- [ ] 151. KV Cache
- [ ] 152. KV Cache Memory Cost

### 7.2 Attention Variants for Efficiency
- [ ] 153. Multi-Query Attention (MQA) — Shazeer, 2019
- [ ] 154. Grouped-Query Attention (GQA) — Ainslie et al., 2023
- [ ] 155. Sliding Window Attention — Longformer, 2020
- [ ] 156. Local Attention
- [ ] 157. Global Attention (special tokens)
- [ ] 158. Sparse Attention — BigBird, 2020
- [ ] 159. Strided/Dilated Attention
- [ ] 160. Random Attention
- [ ] 161. Linear Attention — Performer, 2020

### 7.3 Flash Attention
- [ ] 162. IO-Aware Algorithm Design
- [ ] 163. Tiling (blocking for memory)
- [ ] 164. Online Softmax
- [ ] 165. **Flash Attention — Dao et al., 2022**
- [ ] 166. Flash Attention 2 — Dao, 2023
- [ ] 167. Flash Attention 3 — 2024

### 7.4 Advanced Attention Innovations
- [ ] 168. Paged Attention — vLLM, 2023
- [ ] 169. Multi-head Latent Attention (MLA) — DeepSeek, 2024
- [ ] 170. Differential Attention — Microsoft, Oct 2024
- [ ] 171. Native Sparse Attention (NSA) — DeepSeek, Feb 2025
- [ ] 172. Flash Sparse Attention (FSA)
- [ ] 173. MoBA (Mixture of Block Attention) — Moonshot

---

## PART 8: MIXTURE OF EXPERTS (MoE)

### 8.1 Core MoE Concepts
- [ ] 174. Expert Network
- [ ] 175. Router Network (Gating)
- [ ] 176. Top-k Routing
- [ ] 177. Sparse Activation
- [ ] 178. Expert Parallelism (EP)
- [ ] 179. All-to-All Communication
- [ ] 180. **Mixture of Experts (original) — Shazeer et al., 2017**
- [ ] 181. Load Balancing Loss (Auxiliary Loss)
- [ ] 182. Capacity Factor
- [ ] 183. Token Dropping
- [ ] 184. Expert Choice Routing

### 8.2 Transformer MoE
- [ ] 185. GShard — Lepikhin et al., Google, 2020
- [ ] 186. Switch Transformer — Fedus et al., Google, 2021
- [ ] 187. Router Z-loss

### 8.3 Advanced MoE (DeepSeek Innovations)
- [ ] 188. Fine-grained Experts
- [ ] 189. Shared Experts
- [ ] 190. DeepSeekMoE Architecture — DeepSeek, 2024
- [ ] 191. Auxiliary-loss-free Load Balancing
- [ ] 192. Global-batch Load Balancing Loss

---

## PART 9: ALTERNATIVE ARCHITECTURES

### 9.1 State Space Models (SSMs)
- [ ] 193. State Space Model (continuous)
- [ ] 194. Discretization
- [ ] 195. S4 (Structured State Space) — Gu et al., 2021
- [ ] 196. Selective Mechanism
- [ ] 197. **Mamba — Gu & Dao, 2023**
- [ ] 198. Mamba-2 — 2024

### 9.2 Extended LSTMs
- [ ] 199. xLSTM — Beck et al., 2024
- [ ] 200. xLSTM 7B — Mar 2025

### 9.3 RWKV
- [ ] 201. RWKV Architecture
- [ ] 202. RWKV-7 "Goose" — Mar 2025
- [ ] 203. RWKV-X

### 9.4 Linear Attention Variants
- [ ] 204. Gated Linear Attention (GLA)
- [ ] 205. DeltaNet
- [ ] 206. Gated DeltaNet

### 9.5 Hybrid Architectures
- [ ] 207. **Jamba — AI21, 2024**
- [ ] 208. Samba
- [ ] 209. Griffin — Google, 2024
- [ ] 210. Zamba

---

## PART 10: BEYOND-TOKEN ARCHITECTURES

### 10.1 Byte-Level Models
- [ ] 211. Byte-level Language Modeling
- [ ] 212. Entropy-based Patching
- [ ] 213. **Byte Latent Transformer (BLT) — Meta, Dec 2024**

### 10.2 Sentence-Level Models
- [ ] 214. Sentence Embeddings
- [ ] 215. SONAR Embeddings — Meta
- [ ] 216. **Large Concept Model (LCM) — Meta, Dec 2024**
- [ ] 217. SONAR-LLM — Aug 2025

---

## PART 11: DECODING & INFERENCE

### 11.1 Decoding Strategies
- [ ] 218. Greedy Decoding
- [ ] 219. Temperature Sampling
- [x] 220. Top-k Sampling
- [ ] 221. Top-p (Nucleus) Sampling
- [ ] 222. Min-p Sampling — 2023
- [ ] 223. Typical Sampling
- [ ] 224. Mirostat
- [ ] 225. Beam Search
- [ ] 226. Diverse Beam Search
- [ ] 227. Repetition Penalty
- [ ] 228. Presence Penalty
- [ ] 229. Frequency Penalty
- [ ] 230. Contrastive Decoding — 2022
- [ ] 231. Classifier-Free Guidance (for text) — 2024

### 11.2 Speculative Decoding
- [ ] 232. Draft Model
- [ ] 233. Verification (parallel)
- [ ] 234. **Speculative Decoding — 2023**
- [ ] 235. Self-Speculation
- [ ] 236. Medusa Heads
- [ ] 237. EAGLE
- [ ] 238. Lookahead Decoding

### 11.3 Batching & Serving
- [ ] 239. Static Batching
- [ ] 240. Dynamic Batching
- [ ] 241. Continuous Batching
- [ ] 242. Chunked Prefill
- [ ] 243. Prefix Caching
- [ ] 244. Context Caching — Kimi
- [ ] 245. Disaggregated Serving — Mooncake/Kimi

### 11.4 KV Cache Optimization
- [ ] 246. KV Cache Eviction
- [ ] 247. H2O (Heavy-Hitter Oracle)
- [ ] 248. SnapKV
- [ ] 249. PyramidKV
- [ ] 250. StreamingLLM

### 11.5 Inference Frameworks
- [ ] 251. vLLM
- [ ] 252. SGLang
- [ ] 253. TensorRT-LLM
- [ ] 254. Text Generation Inference (TGI)
- [ ] 255. llama.cpp
- [ ] 256. GGML/GGUF Format

---

## PART 12: QUANTIZATION

### 12.1 Post-Training Quantization
- [ ] 257. INT8 Quantization
- [ ] 258. LLM.int8() — Dettmers et al., 2022
- [ ] 259. 4-bit Quantization
- [ ] 260. GPTQ — Frantar et al., 2022
- [ ] 261. AWQ (Activation-aware Weight Quantization) — 2023
- [ ] 262. QuIP# — 2023
- [ ] 263. EXL2
- [ ] 264. HQQ
- [ ] 265. AQLM

### 12.2 Quantization-Aware Training
- [ ] 266. INT4 Native Training — Kimi K2

---

## PART 13: POST-TRAINING — SUPERVISED FINE-TUNING

### 13.1 Instruction Tuning
- [ ] 267. Instruction Following
- [ ] 268. Instruction-Response Pairs
- [ ] 269. **Supervised Fine-Tuning (SFT)**
- [ ] 270. Multi-turn Conversation Tuning
- [ ] 271. Chat Template
- [ ] 272. System Prompt

### 13.2 Parameter-Efficient Fine-Tuning (PEFT)
- [ ] 273. Adapter Layers — Houlsby et al., 2019
- [ ] 274. Low-Rank Decomposition
- [ ] 275. **LoRA (Low-Rank Adaptation) — Hu et al., 2021**
- [ ] 276. LoRA Rank
- [ ] 277. LoRA Alpha
- [ ] 278. **QLoRA — Dettmers et al., 2023**
- [ ] 279. Double Quantization
- [ ] 280. NormalFloat (NF4)
- [ ] 281. DoRA (Weight-Decomposed LoRA) — 2024

### 13.3 Data for SFT
- [ ] 282. Evol-Instruct — WizardLM
- [ ] 283. Self-Instruct
- [ ] 284. Rejection Sampling
- [ ] 285. Cold-Start Data (for reasoning)

---

## PART 14: POST-TRAINING — REWARD MODELING

### 14.1 Human Preferences
- [ ] 286. Pairwise Preference Data
- [ ] 287. Bradley-Terry Model
- [ ] 288. **Reward Model (RM)**
- [ ] 289. Reward Model Loss

### 14.2 Process vs Outcome Rewards
- [ ] 290. Outcome Reward Model (ORM)
- [ ] 291. Process Reward Model (PRM)
- [ ] 292. Step-level Annotations

### 14.3 Rule-Based Rewards
- [ ] 293. Correctness Verification (math, code)
- [ ] 294. Format Reward
- [ ] 295. Self-Critique Rubric Reward — Kimi

---

## PART 15: POST-TRAINING — REINFORCEMENT LEARNING

### 15.1 RL Foundations
- [ ] 296. Policy (π)
- [ ] 297. Reward Signal
- [ ] 298. Value Function
- [ ] 299. Advantage Function
- [ ] 300. Policy Gradient

### 15.2 PPO
- [ ] 301. TRPO (Trust Region Policy Optimization)
- [ ] 302. Clipping (policy ratio)
- [ ] 303. **PPO (Proximal Policy Optimization) — Schulman et al., 2017**
- [ ] 304. Critic Model (Value Network)
- [ ] 305. GAE (Generalized Advantage Estimation)

### 15.3 RLHF
- [ ] 306. Reference Policy (π_ref)
- [ ] 307. KL Divergence Penalty
- [ ] 308. KL Coefficient (β)
- [ ] 309. **RLHF (Reinforcement Learning from Human Feedback) — Ouyang et al., 2022**
- [ ] 310. **InstructGPT — OpenAI, 2022**
- [ ] 311. Reward Hacking

### 15.4 RLAIF & Constitutional AI
- [ ] 312. AI Feedback
- [ ] 313. **RLAIF (RL from AI Feedback)**
- [ ] 314. Constitution (principles)
- [ ] 315. **Constitutional AI (CAI) — Anthropic, 2022**

---

## PART 16: POST-TRAINING — DIRECT PREFERENCE OPTIMIZATION

### 16.1 DPO Family
- [ ] 316. Reward Reparameterization
- [ ] 317. **DPO (Direct Preference Optimization) — Rafailov et al., 2023**
- [ ] 318. DPO Loss
- [ ] 319. **IPO (Identity Preference Optimization) — 2023**
- [ ] 320. **KTO (Kahneman-Tversky Optimization) — 2024**
- [ ] 321. **ORPO (Odds Ratio Preference Optimization) — 2024**
- [ ] 322. **SimPO (Simple Preference Optimization) — 2024**

### 16.2 GRPO Family
- [ ] 323. Group Sampling
- [ ] 324. Group Baseline
- [ ] 325. **GRPO (Group Relative Policy Optimization) — DeepSeek, 2024**
- [ ] 326. **DAPO (Decoupled Alignment PO) — 2025**
- [ ] 327. Token-level Loss
- [ ] 328. Length-based Reward Shaping
- [ ] 329. **Dr. GRPO — 2025**

---

## PART 17: REASONING & TEST-TIME COMPUTE

### 17.1 Chain-of-Thought
- [ ] 330. Chain-of-Thought (CoT) — Wei et al., 2022
- [ ] 331. Zero-shot CoT
- [ ] 332. Few-shot CoT
- [ ] 333. Self-Consistency — Wang et al., 2022

### 17.2 Test-Time Scaling
- [ ] 334. Test-Time Compute
- [ ] 335. Sequential Scaling
- [ ] 336. Parallel Scaling
- [ ] 337. Best-of-N Sampling
- [ ] 338. Weighted Voting
- [ ] 339. Budget Forcing
- [ ] 340. Thinking Tokens

### 17.3 Search Methods
- [ ] 341. Tree-of-Thought — Yao et al., 2023
- [ ] 342. Monte Carlo Tree Search (MCTS)
- [ ] 343. Verifier-guided Decoding

### 17.4 Latent Reasoning
- [ ] 344. Latent Space (representation)
- [ ] 345. **Coconut (Chain of Continuous Thought) — Dec 2024**
- [ ] 346. Breadth-First Search (BFS) in latent space
- [ ] 347. CCoT / HCoT
- [ ] 348. LightThinker
- [ ] 349. SoftCoT

### 17.5 Self-Improvement
- [ ] 350. Self-Correction
- [ ] 351. Self-Refinement
- [ ] 352. Self-Verification

### 17.6 Thinking Modes
- [ ] 353. Thinking Mode vs Non-Thinking Mode
- [ ] 354. Interleaved Thinking

---

## PART 18: REINFORCEMENT LEARNING WITH VERIFIABLE REWARDS

### 18.1 RLVR Paradigm
- [ ] 355. Verifiable Tasks (math, code)
- [ ] 356. Rule-based Reward Function
- [ ] 357. **RLVR (RL with Verifiable Rewards)**
- [ ] 358. No SFT Before RL (R1-Zero approach)

### 18.2 Emergent Behaviors
- [ ] 359. Emergent Self-Reflection
- [ ] 360. Emergent Verification
- [ ] 361. "Aha Moments"
- [ ] 362. Increasing Response Length for Harder Problems

### 18.3 Distillation from Reasoning
- [ ] 363. Reasoning Trace Distillation
- [ ] 364. Chain-of-Thought Distillation

---

## PART 19: ADVANCED RL FOR PRE-TRAINING

### 19.1 RL on Pre-training Data
- [ ] 365. **Reinforcement Pre-Training (RPT) — Jun 2025**
- [ ] 366. **RLPT — Sep 2025**
- [ ] 367. Autoregressive Segment Reasoning (ASR)
- [ ] 368. Middle Segment Reasoning (MSR)

### 19.2 Test-Time RL
- [ ] 369. **Test-Time Reinforcement Learning (TTRL) — Apr 2025**
- [ ] 370. Majority Voting as Reward Signal

---

## PART 20: DIFFUSION LANGUAGE MODELS

### 20.1 Diffusion Foundations (for LLMs)
- [ ] 371. Forward Diffusion Process
- [ ] 372. Reverse Denoising Process
- [ ] 373. Masked Diffusion

### 20.2 Diffusion LLMs
- [ ] 374. **LLaDA — Feb 2025**
- [ ] 375. Reversal Curse (addressed)
- [ ] 376. Bidirectional Dependencies
- [ ] 377. **Dream 7B — Aug 2025**
- [ ] 378. MDLM (Masked Diffusion LM)
- [ ] 379. Block Diffusion
- [ ] 380. **d1 — Apr 2025**
- [ ] 381. Mercury Coder — Inception Labs, 2025

---

## PART 21: MULTI-TOKEN PREDICTION

- [ ] 382. Multi-Token Prediction (concept)
- [ ] 383. MTP Modules (sequential heads)
- [ ] 384. Causal Chain Preservation
- [ ] 385. Shared Output Head
- [ ] 386. **Multi-Token Prediction — DeepSeek-V3, 2024**
- [ ] 387. MTP for Speculative Decoding

---

## PART 22: MEMORY & RETRIEVAL

### 22.1 Memory Mechanisms
- [ ] 388. Memory Tokens
- [ ] 389. Landmark Tokens
- [ ] 390. Compressive Memory
- [ ] 391. Recurrent Memory Transformer

### 22.2 Neural Memory
- [ ] 392. Neural Long-term Memory
- [ ] 393. Memory as Context (MAC)
- [ ] 394. Memory as Gate (MAG)
- [ ] 395. Memory as Layer (MAL)
- [ ] 396. "Surprise" Metric for Memorization
- [ ] 397. Gradient-based Memory Updates
- [ ] 398. **Titans — Google, Dec 2024**

### 22.3 Retrieval Augmented Generation
- [ ] 399. Dense Retrieval
- [ ] 400. Sparse Retrieval (BM25)
- [ ] 401. Hybrid Retrieval
- [ ] 402. Reranking
- [ ] 403. Late Interaction (ColBERT)
- [ ] 404. HyDE (Hypothetical Document Embeddings)
- [ ] 405. **RAG (Retrieval Augmented Generation)**

---

## PART 23: CONTEXT LENGTH EXTENSION (EXPANDED)

- [ ] 406. Ring Attention — 2023
- [ ] 407. Landmark Attention
- [ ] 408. LM-Infinite
- [ ] 409. Dynamic NTK
- [ ] 410. Unlimiformer
- [ ] 411. Memorizing Transformers
- [ ] 412. ∞-former
- [ ] 413. Focused Transformer

---

## PART 24: MULTIMODAL

### 24.1 Vision Encoders
- [ ] 414. Vision Transformer (ViT) — Dosovitskiy et al., 2020
- [ ] 415. CLIP — OpenAI, 2021
- [ ] 416. SigLIP
- [ ] 417. SigLIP-2
- [ ] 418. EVA-CLIP
- [ ] 419. DINOv2
- [ ] 420. InternViT

### 24.2 Vision-Language Connectors
- [ ] 421. Linear Projection (vision to LLM)
- [ ] 422. MLP Projection
- [ ] 423. Perceiver Resampler
- [ ] 424. Q-Former — BLIP-2
- [ ] 425. C-Abstractor
- [ ] 426. D-Abstractor
- [ ] 427. Cross-attention Layers (for multimodal)

### 24.3 Fusion Strategies
- [ ] 428. Late Fusion
- [ ] 429. Mid Fusion
- [ ] 430. Early Fusion (Native Multimodal)

### 24.4 Image Handling
- [ ] 431. Patch-based Tokenization
- [ ] 432. Dynamic Resolution
- [ ] 433. Aspect Ratio Bucketing
- [ ] 434. Tile-based Processing
- [ ] 435. Any-resolution Support

### 24.5 Video Handling
- [ ] 436. Frame Sampling Strategies
- [ ] 437. Temporal Encoding
- [ ] 438. T-RoPE (Temporal RoPE)
- [ ] 439. Text-based Timestamp Alignment
- [ ] 440. Interleaved M-RoPE — Qwen3-VL

### 24.6 Multimodal Architectures
- [ ] 441. DeepStack — Qwen3-VL
- [ ] 442. Dual-Tower (X-Fusion)
- [ ] 443. Native VLM (NEO)

### 24.7 Audio
- [ ] 444. Whisper-style Encoder
- [ ] 445. Speech Tokenization
- [ ] 446. Audio-Text Alignment

---

## PART 25: MODEL MERGING & CONVERSION

### 25.1 Merging Methods
- [ ] 447. Model Averaging (Soup)
- [ ] 448. SLERP (Spherical Linear Interpolation)
- [ ] 449. TIES-Merging
- [ ] 450. DARE (Drop and Rescale)
- [ ] 451. Task Arithmetic
- [ ] 452. Fisher Merging

### 25.2 Architecture Conversion
- [ ] 453. Transformer to RNN Conversion
- [ ] 454. T2R
- [ ] 455. SUPRA
- [ ] 456. RADLADS
- [ ] 457. Llamba
- [ ] 458. LOLCats

---

## PART 26: ADVANCED ATTENTION (2025)

- [ ] 459. PaTH Attention — MIT, Dec 2025
- [ ] 460. Forgetting Transformer (FoX)

---

## PART 27: RECURSIVE & HIERARCHICAL MODELS

- [ ] 461. Hierarchical Reasoning Model (HRM)
- [ ] 462. Thinking Recursive Model (TRM)
- [ ] 463. Recursive Language Model (RLM) — Oct 2025

---

## PART 28: KEY MODEL RELEASES (Chronological)

### 2017
- [ ] 464. **Transformer — Vaswani et al., Jun 2017**

### 2018
- [ ] 465. **GPT-1 — OpenAI, Jun 2018**
- [ ] 466. **BERT — Google, Oct 2018**

### 2019
- [ ] 467. **GPT-2 — OpenAI, Feb 2019**
- [ ] 468. **Transformer-XL — Dai et al., Jan 2019**
- [ ] 469. **RMSNorm — Zhang & Sennrich, 2019**
- [ ] 470. **MQA — Shazeer, 2019**

### 2020
- [ ] 471. **GPT-3 — OpenAI, May 2020**
- [ ] 472. **T5 — Google, 2020**
- [ ] 473. **Longformer — 2020**
- [ ] 474. **BigBird — 2020**
- [ ] 475. **GShard — Google, 2020**

### 2021
- [ ] 476. **Switch Transformer — Google, Jan 2021**
- [ ] 477. **RoPE — Su et al., 2021**
- [ ] 478. **ALiBi — Press et al., 2021**
- [ ] 479. **LoRA — Hu et al., 2021**
- [ ] 480. **CLIP — OpenAI, 2021**

### 2022
- [ ] 481. **InstructGPT — OpenAI, Mar 2022**
- [ ] 482. **Constitutional AI — Anthropic, Dec 2022**
- [ ] 483. **Flash Attention — Dao, May 2022**
- [ ] 484. **Chain-of-Thought — Wei et al., Jan 2022**
- [ ] 485. **Self-Consistency — Wang et al., 2022**
- [ ] 486. **GPTQ — Oct 2022**
- [ ] 487. **LLM.int8() — Aug 2022**
- [ ] 488. **DeepNorm — 2022**

### 2023
- [ ] 489. **LLaMA — Meta, Feb 2023**
- [ ] 490. **LLaMA 2 — Meta, Jul 2023**
- [ ] 491. **Mistral 7B — Mistral, Sep 2023**
- [ ] 492. **Mixtral 8x7B — Mistral, Dec 2023**
- [ ] 493. **Mamba — Gu & Dao, Dec 2023**
- [ ] 494. **DPO — Rafailov et al., May 2023**
- [ ] 495. **QLoRA — Dettmers et al., May 2023**
- [ ] 496. **Flash Attention 2 — Jul 2023**
- [ ] 497. **GQA Paper — Ainslie et al., 2023**
- [ ] 498. **vLLM/Paged Attention — Jun 2023**
- [ ] 499. **Position Interpolation — Chen et al., 2023**
- [ ] 500. **NTK-aware Scaling — 2023**
- [ ] 501. **YaRN — Peng et al., 2023**
- [ ] 502. **Speculative Decoding — 2023**
- [ ] 503. **Tree-of-Thought — Yao et al., 2023**
- [ ] 504. **AWQ — 2023**
- [ ] 505. **Ring Attention — 2023**

### 2024
- [ ] 506. **Llama 3 — Meta, Apr 2024**
- [ ] 507. **Llama 3.1 — Meta, Jul 2024**
- [ ] 508. **Llama 3.2 — Meta, Sep 2024**
- [ ] 509. **Qwen2.5 — Alibaba, Sep 2024**
- [ ] 510. **DeepSeek-V2 — May 2024**
- [ ] 511. **DeepSeek-V3 — Dec 2024**
- [ ] 512. **Jamba — AI21, Mar 2024**
- [ ] 513. **Mamba-2 — May 2024**
- [ ] 514. **Flash Attention 3 — 2024**
- [ ] 515. **LongRoPE — 2024**
- [ ] 516. **Differential Attention — Microsoft, Oct 2024**
- [ ] 517. **Titans — Google, Dec 2024**
- [ ] 518. **Byte Latent Transformer — Meta, Dec 2024**
- [ ] 519. **Large Concept Model — Meta, Dec 2024**
- [ ] 520. **Coconut — Dec 2024**
- [ ] 521. **IPO — 2023**
- [ ] 522. **KTO — 2024**
- [ ] 523. **ORPO — 2024**
- [ ] 524. **SimPO — 2024**
- [ ] 525. **GRPO — DeepSeek, 2024**
- [ ] 526. **DoRA — 2024**
- [ ] 527. **Griffin — Google, 2024**
- [ ] 528. **xLSTM — 2024**

### 2025
- [ ] 529. **DeepSeek R1 — Jan 2025**
- [ ] 530. **Kimi K1.5 — Moonshot, Jan 2025**
- [ ] 531. **Qwen3 — Alibaba, May 2025**
- [ ] 532. **Kimi K2 — Moonshot, Jul 2025**
- [ ] 533. **Llama 4 — Meta, 2025**
- [ ] 534. **Qwen3-Next — Sep 2025**
- [ ] 535. **Qwen3-VL — Nov 2025**
- [ ] 536. **Kimi K2 Thinking — Nov 2025**
- [ ] 537. **LLaDA — Feb 2025**
- [ ] 538. **xLSTM 7B — Mar 2025**
- [ ] 539. **RWKV-7 "Goose" — Mar 2025**
- [ ] 540. **Native Sparse Attention — DeepSeek, Feb 2025**
- [ ] 541. **TTRL — Apr 2025**
- [ ] 542. **d1 — Apr 2025**
- [ ] 543. **RPT — Jun 2025**
- [ ] 544. **Dream 7B — Aug 2025**
- [ ] 545. **SONAR-LLM — Aug 2025**
- [ ] 546. **RLPT — Sep 2025**
- [ ] 547. **Recursive Language Model — Oct 2025**
- [ ] 548. **PaTH Attention — MIT, Dec 2025**
- [ ] 549. **DAPO — 2025**
- [ ] 550. **Dr. GRPO — 2025**

---

## READING ORDER SUMMARY

- [ ] 1. Start with **Part 1-2**: Math foundations, embeddings, RNNs, LSTMs, attention origins
- [ ] 2. Then **Part 3-4**: Transformer architecture, tokenization
- [ ] 3. Then **Part 5-6**: Pre-training paradigms, training infrastructure
- [ ] 4. Then **Part 7-8**: Attention efficiency, MoE
- [ ] 5. Then **Part 9-11**: Alternative architectures, beyond-token models, inference
- [ ] 6. Then **Part 12-16**: Quantization, SFT, reward modeling, RLHF, DPO
- [ ] 7. Then **Part 17-21**: Reasoning, test-time compute, RLVR, diffusion LLMs, MTP
- [ ] 8. Then **Part 22-27**: Memory, context extension, multimodal, merging, advanced topics
- [ ] 9. Finally **Part 28**: Read model papers in chronological order

---