import { NextRequest, NextResponse } from 'next/server';
import { GaussianNB } from 'ml-naivebayes';
import { getDatasetById } from '@/lib/datasets';
import {
    extractFeatures,
    extractLabels,
    encodeLabels,
    trainTestSplit,
    confusionMatrix,
    accuracy,
    precision,
    recall
} from '@/lib/ml-utils';

export async function POST(request: NextRequest) {
    try {
        const body = await request.json();
        const { datasetId, targetColumn, trainRatio = 0.8 } = body;

        // Get dataset
        const dataset = getDatasetById(datasetId);
        if (!dataset) {
            return NextResponse.json(
                { success: false, error: 'Dataset not found' },
                { status: 404 }
            );
        }

        // Use provided target column or default
        const target = targetColumn || dataset.targetColumn;
        if (!target) {
            return NextResponse.json(
                { success: false, error: 'No target column specified' },
                { status: 400 }
            );
        }

        // Extract features and labels
        const features = extractFeatures(dataset.data, dataset.features);
        const labels = extractLabels(dataset.data, target);
        const { encoded: encodedLabels, mapping: labelMapping } = encodeLabels(labels);

        // Shuffle and split data manually
        const n = features.length;
        const shuffledIndices = Array.from({ length: n }, (_, i) => i)
            .sort(() => Math.random() - 0.5);

        const trainSize = Math.floor(n * trainRatio);
        const trainIndices = shuffledIndices.slice(0, trainSize);
        const testIndices = shuffledIndices.slice(trainSize);

        // Ensure we have test data
        if (testIndices.length === 0) {
            return NextResponse.json(
                { success: false, error: 'Not enough data for testing. Try increasing train ratio.' },
                { status: 400 }
            );
        }

        const trainFeatures = trainIndices.map(i => features[i]);
        const trainLabels = trainIndices.map(i => encodedLabels[i]);
        const testFeatures = testIndices.map(i => features[i]);
        const testLabels = testIndices.map(i => encodedLabels[i]);

        // Train Naive Bayes
        const classifier = new GaussianNB();
        classifier.train(trainFeatures, trainLabels);

        // Predict on test set - predict one at a time to avoid dimension issues
        const predictions: number[] = [];
        for (const testRow of testFeatures) {
            const pred = classifier.predict([testRow]);
            predictions.push(pred[0] as number);
        }

        // Calculate metrics
        const confMatrix = confusionMatrix(testLabels, predictions as number[]);
        const acc = accuracy(confMatrix);
        const prec = precision(confMatrix);
        const rec = recall(confMatrix);

        // Reverse label mapping for display
        const classNames = Array.from(labelMapping.entries())
            .sort((a, b) => a[1] - b[1])
            .map(([name]) => String(name));

        return NextResponse.json({
            success: true,
            data: {
                algorithm: 'Naive Bayes (Gaussian)',
                parameters: { targetColumn: target, trainRatio },
                results: {
                    accuracy: Number((acc * 100).toFixed(2)),
                    precision: prec.map(p => Number((p * 100).toFixed(2))),
                    recall: rec.map(r => Number((r * 100).toFixed(2))),
                    confusionMatrix: confMatrix,
                    classNames,
                    trainSize: trainIndices.length,
                    testSize: testIndices.length,
                },
                datasetInfo: {
                    name: dataset.name,
                    rows: dataset.rows,
                    features: dataset.features,
                    targetColumn: target,
                },
            },
        });
    } catch (error) {
        console.error('Naive Bayes error:', error);
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        const errorStack = error instanceof Error ? error.stack : '';
        console.error('Error stack:', errorStack);
        return NextResponse.json(
            { success: false, error: `Naive Bayes failed: ${errorMessage}` },
            { status: 500 }
        );
    }
}
