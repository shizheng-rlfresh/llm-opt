<script>
    import * as tf from '@tensorflow/tfjs';
    import { logisticRegression } from '$lib/tfjs/function';

    const x = tf.tensor([
        [1.0, 0.5, 0.2],
        [0.2, -0.2, -0.4],
    ]);
    const y = tf.tensor([[1.0], [-1.0]]);

    const logReg = logisticRegression({ dim: x.shape[1], bias: true });
    let loss = logReg.computeLoss(x, y, 0.5);
    // loss.print();
    let grad = logReg.computeGrad(x, y, 0.5);
</script>

<p>{loss.dataSync()}</p>
<!-- <p>{grad.wGrad}</p>
<p>{grad.bGrad}</p> -->
