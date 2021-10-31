# Deep-Unfolding-Beamforming-for-Intelligent-Reflecting-Surface-assisted-Full-Duplex-Systems
This project includes codes for paper "Deep-Unfolding-Beamforming-for-Intelligent-Reflecting-Surface-assisted-Full-Duplex-Systems".

1) parameters_config: initialize parameters like K, L, N_r, N_t

2) my_rician_channel: generate channels based on the Rician model [1]

[1] M.-M. Zhao, Q. Wu, M.-J. Zhao, and R. Zhang, ``Intelligent reflecting surface enhanced wireless networks: Two-timescale beamforming optimization,"

3) my_model: The LABN model

4) UnfoldingWithStochasticTheta: inherit from my_model and implement the LPBN

5) DeepUnfolding_Theta: Trains the developed deep-unfolding NN

6) operators: developed libary for complex matrices oprerations such as inverse and multiply.
