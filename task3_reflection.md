Scalar quantization only led to a minor decrease in precision (100% to 99.8%), with a speed gain of about 21% when 
rescoring was set to true, and about 24% when rescoring was set to false.
Since rescoring to false led to a significant drop in precision to 82.9%, the minor speed gain is probably not worth it
in this particular case.

### Averaged results from three runs each:

|                               | precision | avg_time [ms} |
|-------------------------------|-----------|---------------|
| exact k-NN query              | 1.000     | 101.81        |
| ANN, no quantization          | 1.000     | 7.63          |
| ANN, quantized, restore=true  | 0.9980    | 6.04          |
| ANN, quantized, restore=false | 0.8290    | 5.78          |




