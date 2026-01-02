import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
    try {
        const body = await request.json();
        const { results, algorithm, parameters, datasetInfo } = body;

        if (!results) {
            return NextResponse.json(
                { success: false, error: 'No results provided' },
                { status: 400 }
            );
        }

        // Create export object
        const exportData = {
            exportedAt: new Date().toISOString(),
            algorithm,
            parameters,
            dataset: datasetInfo,
            results,
            metadata: {
                version: '1.0',
                generator: 'ALLgorithm',
            },
        };

        return NextResponse.json({
            success: true,
            data: exportData,
        });
    } catch (error) {
        console.error('Export error:', error);
        return NextResponse.json(
            { success: false, error: 'Failed to export results' },
            { status: 500 }
        );
    }
}
