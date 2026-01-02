// Type declarations for ml-naivebayes
declare module 'ml-naivebayes' {
    export class GaussianNB {
        constructor();
        train(features: number[][], labels: number[]): void;
        predict(features: number[][]): number[];
    }
}

// Type declarations for ml-pca
declare module 'ml-pca' {
    interface PCAOptions {
        center?: boolean;
        scale?: boolean;
    }

    interface PredictOptions {
        nComponents?: number;
    }

    interface Matrix {
        data: number[][];
    }

    export class PCA {
        constructor(data: number[][], options?: PCAOptions);
        getExplainedVariance(): number[];
        getCumulativeVariance(): number[];
        getLoadings(): Matrix;
        predict(data: number[][], options?: PredictOptions): Matrix;
    }
}
