# Fast Writer Adaptation

This code is part of the implementation of the handwritten text recognition experiments in the paper [Fast writer adaptation with style extractor network for handwritten text recognition](https://sciencedirect.53yu.com/science/article/abs/pii/S0893608021004755). The code is developed based on the Pytorch framework.

# Highlights

For the HETR task:

(1)If you only need the backbone recognition network: 

![](https://github.com/Wukong90/Handwritten-Text-Recognition/blob/main/imgs/baselines.jpg)

，just run the train_CTC_HAM_Vis_Contex.py;

(2)Train the writer style extractor network：

![](https://github.com/Wukong90/Handwritten-Text-Recognition/blob/main/imgs/SEN.jpg)

, run the train_WID.py;

# Citation

If you use our code in your research or wish to refer to the baseline results, please use the following BibTeX entry.

@article{wang2022fast,  
        title={Fast writer adaptation with style extractor network for handwritten text recognition},  
        author={Wang, Zi-Rui and Du, Jun},  
        journal={Neural Networks},  
        volume={147},  
        pages={42--52},  
        year={2022},   
        publisher={Elsevier}  
}
