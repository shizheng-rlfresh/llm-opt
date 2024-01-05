import * as tf from '@tensorflow/tfjs';

function xyWb(x, y, w, b) {
    return y.mul(x.dot(w).add(b));
}

function expLogReg(xyWb) {
    return tf.exp(xyWb);
}

function weightGradient(x, y, w, b, lambda) {
    return tf.tidy(() => {
        const expPart = expLogReg(xyWb(x, y.neg(), w, b));
        const sigmoidPart = tf.sigmoid(xyWb(x, y, w, b));
        const dataPart = y.neg().mul(x);
        const lambdaPart = w.mul(lambda);

        return tf
            .mean(expPart.mul(sigmoidPart).mul(dataPart).add(lambdaPart.transpose()), 0, true)
            .transpose();
    });
}

function biasGradient(x, y, w, b, lambda) {
    return tf.tidy(() => {
        const expPart = expLogReg(xyWb(x, y.neg(), w, b));
        const sigmoidPart = tf.sigmoid(xyWb(x, y, w, b));
        const dataPart = y.neg();
        const lambdaPart = b.mul(lambda);

        return tf.mean(expPart.mul(sigmoidPart).mul(dataPart).add(lambdaPart), 0).transpose();
    });
}

export function logisticRegression({ dim, bias = true, seed = 46 }) {
    const w = tf.variable(tf.randomStandardNormal([dim, 1], 'float32', seed));
    let b = tf.tensor([0]);

    if (bias) {
        b = tf.randomStandardNormal([1], 'float32', seed);
    }

    function computeLoss(x, y, lambda = 0) {
        const loss = tf.tidy(() => {
            const logi = tf.log(
                tf
                    .exp(y.neg().mul(x.dot(w).add(b)))
                    .add(1.0)
                    .add(
                        tf
                            .square(w)
                            .sum()
                            .add(b.pow(2))
                            .mul(lambda / 2)
                    )
            );
            return tf.mean(logi);
        });
        return loss;
    }

    function computeGrad(x, y, lambda = 0) {
        const grad = tf.tidy(() => {
            const wGrad = weightGradient(x, y, w, b, lambda);
            const bGrad = biasGradient(x, y, w, b, lambda);

            return { w: wGrad, b: bGrad };
        });
        return grad;
    }

    function step(opt) {
        opt(w, b);
    }

    function getParams() {
        return { w: w.dataSync(), b: b.dataSync() };
    }

    return { computeLoss, computeGrad };
}
