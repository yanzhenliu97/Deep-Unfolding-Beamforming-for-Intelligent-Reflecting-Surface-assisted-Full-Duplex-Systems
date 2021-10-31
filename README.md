# Deep-Unfolding-Beamforming-for-Intelligent-Reflecting-Surface-assisted-Full-Duplex-Systems
This project includes codes for paper "Deep-Unfolding-Beamforming-for-Intelligent-Reflecting-Surface-assisted-Full-Duplex-Systems". The functions of the codes are described as follows

1) parameters_config: Initialize system parameters like K, L, N_r, N_t

2) my_rician_channel: Generate channels based on the Rician model [1]

   [1] M.-M. Zhao, Q. Wu, M.-J. Zhao, and R. Zhang, ``Intelligent reflecting surface enhanced wireless networks: Two-timescale beamforming optimization,"

3) my_model: The LABN model

4) UnfoldingWithStochasticTheta: Inherit from my_model and implement the LPBN

5) DeepUnfolding_Theta: Generate channels and trains the developed deep-unfolding NN

6) operators: The developed libary for complex matrices oprerations such as inverse and multiply.
