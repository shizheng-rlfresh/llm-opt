import * as tf from '@tensorflow/tfjs';

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
        console.log(w.shape);
        console.log(tf.log(tf.exp(y.neg().mul(x.dot(w).add(b))).add(1.0)).print());
        const grad = 0;

        // const grad = tf.tidy(() => {
        //     const wGrad = tf.tidy(() => {
        //         const logi =
        //             (1 / tf.log(1.0 + tf.exp(-y.mul(x.dot(w).add(b))))) *
        //                 tf.exp(-y.mul(x.dot(w).add(b))) *
        //                 (-y).mul(x.transpose()) +
        //             lambda * w;
        //         return tf.mean(logi, 0).transpose();
        //     });

        //     const bGrad = tf.tidy(() => {
        //         const logi =
        //             (1 / tf.log(1.0 + tf.exp(-y.mul(x.dot(w).add(b))))) *
        //                 tf.exp(-y.mul(x.dot(w).add(b))) *
        //                 -y +
        //             lambda * b;
        //         return tf.mean(logi);
        //     });

        //     return { w: wGrad, b: bGrad };
        // });

        return grad;
    }

    return { computeLoss, computeGrad };
}
