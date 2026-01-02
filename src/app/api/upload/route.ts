import { NextRequest, NextResponse } from 'next/server';
import Papa from 'papaparse';

export async function POST(request: NextRequest) {
    try {
        const formData = await request.formData();
        const file = formData.get('file') as File | null;

        if (!file) {
            return NextResponse.json(
                { success: false, error: 'No file provided' },
                { status: 400 }
            );
        }

        // Check file type
        if (!file.name.endsWith('.csv')) {
            return NextResponse.json(
                { success: false, error: 'Only CSV files are supported' },
                { status: 400 }
            );
        }

        // Check file size (max 5MB)
        if (file.size > 5 * 1024 * 1024) {
            return NextResponse.json(
                { success: false, error: 'File size must be less than 5MB' },
                { status: 400 }
            );
        }

        // Parse CSV
        const text = await file.text();
        const result = Papa.parse(text, {
            header: true,
            skipEmptyLines: true,
            dynamicTyping: true,
        });

        if (result.errors.length > 0) {
            return NextResponse.json(
                { success: false, error: 'Failed to parse CSV: ' + result.errors[0].message },
                { status: 400 }
            );
        }

        const data = result.data as Record<string, unknown>[];
        const fields = result.meta.fields || [];

        // Detect numeric columns
        const numericColumns = fields.filter(field => {
            const values = data.map(row => row[field]);
            return values.every(v => v === null || typeof v === 'number');
        });

        // Generate a unique ID for this upload
        const uploadId = `custom_${Date.now()}`;

        return NextResponse.json({
            success: true,
            data: {
                id: uploadId,
                name: file.name.replace('.csv', ''),
                rows: data.length,
                columns: fields,
                numericColumns,
                preview: data.slice(0, 5),
                fullData: data,
            },
        });
    } catch (error) {
        console.error('Upload error:', error);
        return NextResponse.json(
            { success: false, error: 'Failed to process upload' },
            { status: 500 }
        );
    }
}
