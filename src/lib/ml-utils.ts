// ML Utilities for ALLgorithm
// Helper functions for machine learning operations

/**
 * Normalize data to 0-1 range
 */
export function normalizeData(data: number[][]): number[][] {
    const numFeatures = data[0].length;
    const mins: number[] = [];
    const maxs: number[] = [];

    // Find min/max for each feature
    for (let j = 0; j < numFeatures; j++) {
        const column = data.map(row => row[j]);
        mins.push(Math.min(...column));
        maxs.push(Math.max(...column));
    }

    // Normalize
    return data.map(row =>
        row.map((val, j) => {
            const range = maxs[j] - mins[j];
            return range === 0 ? 0 : (val - mins[j]) / range;
        })
    );
}

/**
 * Calculate Euclidean distance between two points
 */
export function euclideanDistance(a: number[], b: number[]): number {
    return Math.sqrt(a.reduce((sum, val, i) => sum + Math.pow(val - b[i], 2), 0));
}

/**
 * Calculate silhouette score for clustering
 */
export function silhouetteScore(data: number[][], labels: number[]): number {
    const n = data.length;
    if (n < 2) return 0;

    const silhouettes: number[] = [];

    for (let i = 0; i < n; i++) {
        const ownCluster = labels[i];
        const ownClusterPoints = data.filter((_, idx) => labels[idx] === ownCluster && idx !== i);

        if (ownClusterPoints.length === 0) {
            silhouettes.push(0);
            continue;
        }

        // Calculate a(i) - average distance to own cluster
        const a = ownClusterPoints.reduce((sum, p) => sum + euclideanDistance(data[i], p), 0) / ownClusterPoints.length;

        // Calculate b(i) - minimum average distance to other clusters
        const otherClusters = [...new Set(labels)].filter(c => c !== ownCluster);

        if (otherClusters.length === 0) {
            silhouettes.push(0);
            continue;
        }

        const b = Math.min(...otherClusters.map(c => {
            const clusterPoints = data.filter((_, idx) => labels[idx] === c);
            return clusterPoints.reduce((sum, p) => sum + euclideanDistance(data[i], p), 0) / clusterPoints.length;
        }));

        const s = (b - a) / Math.max(a, b);
        silhouettes.push(isNaN(s) ? 0 : s);
    }

    return silhouettes.reduce((a, b) => a + b, 0) / silhouettes.length;
}

/**
 * Calculate inertia (within-cluster sum of squares)
 */
export function calculateInertia(data: number[][], labels: number[], centroids: number[][]): number {
    return data.reduce((sum, point, i) => {
        const centroid = centroids[labels[i]];
        return sum + Math.pow(euclideanDistance(point, centroid), 2);
    }, 0);
}

/**
 * Calculate confusion matrix
 */
export function confusionMatrix(actual: number[], predicted: number[]): number[][] {
    const classes = [...new Set([...actual, ...predicted])].sort((a, b) => a - b);
    const n = classes.length;
    const matrix: number[][] = Array(n).fill(null).map(() => Array(n).fill(0));

    actual.forEach((a, i) => {
        const actualIdx = classes.indexOf(a);
        const predictedIdx = classes.indexOf(predicted[i]);
        matrix[actualIdx][predictedIdx]++;
    });

    return matrix;
}

/**
 * Calculate accuracy from confusion matrix
 */
export function accuracy(matrix: number[][]): number {
    const total = matrix.flat().reduce((a, b) => a + b, 0);
    const correct = matrix.reduce((sum, row, i) => sum + row[i], 0);
    return total === 0 ? 0 : correct / total;
}

/**
 * Calculate precision for each class
 */
export function precision(matrix: number[][]): number[] {
    return matrix[0].map((_, classIdx) => {
        const tp = matrix[classIdx][classIdx];
        const fp = matrix.reduce((sum, row, i) => i !== classIdx ? sum + row[classIdx] : sum, 0);
        return tp + fp === 0 ? 0 : tp / (tp + fp);
    });
}

/**
 * Calculate recall for each class
 */
export function recall(matrix: number[][]): number[] {
    return matrix.map((row, classIdx) => {
        const tp = row[classIdx];
        const fn = row.reduce((sum, val, i) => i !== classIdx ? sum + val : sum, 0);
        return tp + fn === 0 ? 0 : tp / (tp + fn);
    });
}

/**
 * Split data into train and test sets
 */
export function trainTestSplit<T>(data: T[], testRatio: number = 0.2): { train: T[]; test: T[] } {
    const shuffled = [...data].sort(() => Math.random() - 0.5);
    const testSize = Math.floor(data.length * testRatio);
    return {
        test: shuffled.slice(0, testSize),
        train: shuffled.slice(testSize),
    };
}

/**
 * Extract numeric features from dataset
 */
export function extractFeatures(
    data: Record<string, number | string>[],
    featureColumns: string[]
): number[][] {
    return data.map(row =>
        featureColumns.map(col => {
            const val = row[col];
            return typeof val === 'number' ? val : 0;
        })
    );
}

/**
 * Extract labels from dataset
 */
export function extractLabels(
    data: Record<string, number | string>[],
    labelColumn: string
): (number | string)[] {
    return data.map(row => row[labelColumn]);
}

/**
 * Encode categorical labels to numbers
 */
export function encodeLabels(labels: (number | string)[]): { encoded: number[]; mapping: Map<string | number, number> } {
    const unique = [...new Set(labels)];
    const mapping = new Map<string | number, number>();
    unique.forEach((label, idx) => mapping.set(label, idx));

    return {
        encoded: labels.map(l => mapping.get(l)!),
        mapping,
    };
}
