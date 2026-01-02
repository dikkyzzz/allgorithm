import { NextResponse } from 'next/server';
import { getDatasetInfo } from '@/lib/datasets';

export async function GET() {
    try {
        const datasets = getDatasetInfo();
        return NextResponse.json({
            success: true,
            data: datasets
        });
    } catch (error) {
        return NextResponse.json(
            { success: false, error: 'Failed to fetch datasets' },
            { status: 500 }
        );
    }
}
