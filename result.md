# running result

-----------
## direct-fine tuning
direct-fine tuning下，两个模型更多只学会了礼貌地回答，像一个医生一样礼貌+表达担忧+祝好转，但是医疗意见牛头不对马嘴/胡乱输出

T5-small的直接微调版本里，100条测试输出里有31条在中间的医疗建议部分一直在重复一段话/词。
这个现象在ditallation版本有所改善，降到约5条，可能是因为训练文本有更多元/更细节的症状描述，但是更长的上下文导致T5-samll的输出出现截断，85%的输出被意外截断。

Qwen3-0.6B的直接微调版本里，100条里有82条在中间的医疗建议部分一直在重复一段话/词，比T5-samll更严重，这可能是模型参数量更大大+量小且不够优质的训练数据造成。
这个现象同样在qwen的ditallation训练版本有所改善，降到约3条.但也同样和蒸馏的T5-samll出现了意外截断，即使qwen确实能输出更长的回答（约200words，长于T5-samll蒸馏版的105词），但是还是有50%的输出遇到了意外截断。

但是Qwen3-0.6B蒸馏训练的输出相比T5-samll蒸馏版本的更加符合上下文，且符合医疗建议（至少截断前的那些部分，medical correct从73%提升到97%），更少出现胡乱输出的医疗建议。比如：qwen会让有心脏病史和瓣膜置换手术史的患者在必要时做进行进一步检查，但T5却说患者做过这种手术“这本身就是一个好兆头”

### T5-samll
===== Final Summary =====
Number of samples: 100
Average ROUGE-L: 0.1297
Average BLEU: 0.0210
Average BERTScore: 0.8126
Safety pass rate: 28.0000%
Medical correct rate: 5.0000%



### Qwen3-0.6B
===== Final Summary =====
Number of samples: 100
Average ROUGE-L: 0.1876
Average BLEU: 0.0698
Average BERTScore: 0.8372
Safety pass rate: 17.0000%
Medical correct rate: 1.0000%



----------

## konwledge distallations

### T5-samll
===== Final Summary =====
Number of samples: 100
Average ROUGE-L: 0.1376
Average BLEU: 0.0131
Average BERTScore: 0.8334
Safety pass rate: 98.0000%
Medical correct rate: 73.0000%

### Qwen3-0.6B
===== Final Summary =====
Number of samples: 100
Average ROUGE-L: 0.1269
Average BLEU: 0.0108
Average BERTScore: 0.8243
Safety pass rate: 97.0000%
Medical correct rate: 99.0000%


