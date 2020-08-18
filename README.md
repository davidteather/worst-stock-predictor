# Worst-Stock-Predictor

The training of the model is based on [this](https://www.thepythoncode.com/article/stock-price-prediction-in-python-using-tensorflow-2-and-keras) article and their code. However the evaluation script of all stocks on the NYSE and analysis is written by me.

# Demo

[![](thumbnail.png?raw=true)](https://www.youtube.com/watch?v=OEbb5G4Awwc)

## Installing 

Run the following command (tested on python 3.7.3)
```
pip install -r requirements.txt
```

## Training the model

Remove the current pre-trained model in results (or change the stock you want to train it on) and run the following command.
```
python train.py
```

You can change the stock it trains on and a few other options within the train.py file itself. 

## Evaluating the model

If you want to only evaluate one stock at a time and show a graph of it run single_eval.py and change the ticker variable near the top of the file.

If you want to evaluate the model as a whole run eval.py and it will end looping over ~6,500 stocks on the NYSE. After this has finished run analysis.py and it'll give you some information about the data stored in the results.json file.

## Disclaimer

I am not responsible for any losses you may experience while using any models trained by the code I have provided. You should not expect to make any money and expect to lose it all, after all that was the primary focus of this project.