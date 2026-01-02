import { algorithms } from '@/lib/algorithms';
import { algorithmDocs } from '@/lib/algorithm-docs';
import { notFound } from 'next/navigation';
import Link from 'next/link';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Header } from '@/components/landing/Header';
import { BookOpen, Code, CheckCircle2, XCircle, Lightbulb, FileText, ExternalLink, Clock, Database, ArrowRight } from 'lucide-react';

interface PageProps {
    params: Promise<{ id: string }>;
}

export async function generateStaticParams() {
    return algorithms.map((algo) => ({ id: algo.id }));
}

export default async function AlgorithmDocsPage({ params }: PageProps) {
    const { id } = await params;
    const algorithm = algorithms.find((a) => a.id === id);
    const doc = algorithmDocs[id];

    if (!algorithm || !doc) {
        notFound();
    }

    const categoryColors: Record<string, string> = {
        'Clustering': 'bg-blue-50 text-blue-700 border-blue-200',
        'Classification': 'bg-violet-50 text-violet-700 border-violet-200',
        'Regression': 'bg-emerald-50 text-emerald-700 border-emerald-200',
        'Dimensionality Reduction': 'bg-amber-50 text-amber-700 border-amber-200',
        'Text Mining': 'bg-rose-50 text-rose-700 border-rose-200',
    };

    return (
        <div className="min-h-screen bg-gray-50">
            <Header />

            {/* Hero */}
            <div className="bg-[#004040] pt-24">
                <div className="container max-w-5xl mx-auto px-4 py-16">
                    <div className="flex items-center gap-2 mb-4">
                        <Badge className="bg-white/20 text-white border-white/30">
                            Documentation
                        </Badge>
                        <Badge className={categoryColors[algorithm.category]}>
                            {algorithm.category}
                        </Badge>
                    </div>
                    <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
                        {doc.fullName}
                    </h1>
                    <p className="text-xl text-white/80 max-w-3xl">
                        Complete technical documentation with mathematical foundations, implementation details, and practical guidelines.
                    </p>
                </div>
            </div>

            {/* Content */}
            <div className="container max-w-5xl mx-auto px-4 py-12">
                <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
                    {/* Sidebar - Table of Contents */}
                    <div className="lg:col-span-1">
                        <div className="sticky top-24 space-y-4">
                            <Card className="bg-white border-gray-200 shadow-sm">
                                <CardHeader className="pb-3">
                                    <CardTitle className="text-sm text-gray-900">Contents</CardTitle>
                                </CardHeader>
                                <CardContent className="space-y-1">
                                    {[
                                        { id: 'introduction', label: 'Introduction' },
                                        { id: 'math', label: 'Mathematical Background' },
                                        { id: 'algorithm', label: 'Algorithm' },
                                        { id: 'complexity', label: 'Complexity' },
                                        { id: 'guidelines', label: 'Usage Guidelines' },
                                        { id: 'examples', label: 'Examples' },
                                        { id: 'references', label: 'References' },
                                    ].map((item) => (
                                        <a
                                            key={item.id}
                                            href={`#${item.id}`}
                                            className="block text-sm text-gray-600 hover:text-[#004040] py-1 transition-colors"
                                        >
                                            {item.label}
                                        </a>
                                    ))}
                                </CardContent>
                            </Card>

                            <Link href={`/algorithms/${id}`} className="block">
                                <Card className="bg-[#004040]/5 border-[#004040]/20 hover:bg-[#004040]/10 transition-colors">
                                    <CardContent className="p-4 flex items-center justify-between">
                                        <span className="text-sm font-medium text-[#004040]">Try Demo →</span>
                                    </CardContent>
                                </Card>
                            </Link>
                        </div>
                    </div>

                    {/* Main Content */}
                    <div className="lg:col-span-3 space-y-10">
                        {/* Introduction */}
                        <section id="introduction">
                            <h2 className="text-2xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                                <BookOpen className="w-6 h-6 text-[#004040]" />
                                Introduction
                            </h2>
                            <div className="prose prose-gray max-w-none">
                                <p className="text-gray-600 leading-relaxed whitespace-pre-line">
                                    {doc.introduction}
                                </p>
                            </div>
                        </section>

                        {/* Mathematical Background */}
                        <section id="math">
                            <h2 className="text-2xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                                <FileText className="w-6 h-6 text-[#004040]" />
                                Mathematical Background
                            </h2>
                            <div className="space-y-6">
                                {doc.mathematicalBackground.map((item, i) => (
                                    <Card key={i} className="bg-white border-gray-200 shadow-sm">
                                        <CardContent className="p-6">
                                            <h3 className="font-semibold text-gray-900 mb-2">{item.title}</h3>
                                            <p className="text-gray-600 text-sm mb-4">{item.content}</p>
                                            {item.formula && (
                                                <div className="p-4 bg-gray-900 rounded-lg">
                                                    <code className="text-emerald-400 font-mono text-lg">{item.formula}</code>
                                                </div>
                                            )}
                                        </CardContent>
                                    </Card>
                                ))}
                            </div>
                        </section>

                        {/* Algorithm / Pseudocode */}
                        <section id="algorithm">
                            <h2 className="text-2xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                                <Code className="w-6 h-6 text-[#004040]" />
                                Algorithm (Pseudocode)
                            </h2>
                            <Card className="bg-gray-900 border-gray-800">
                                <CardContent className="p-6">
                                    <pre className="text-gray-100 font-mono text-sm whitespace-pre-wrap leading-relaxed">
                                        {doc.pseudocode}
                                    </pre>
                                </CardContent>
                            </Card>
                        </section>

                        {/* Complexity */}
                        <section id="complexity">
                            <h2 className="text-2xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                                <Clock className="w-6 h-6 text-[#004040]" />
                                Computational Complexity
                            </h2>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <Card className="bg-white border-gray-200 shadow-sm">
                                    <CardContent className="p-6">
                                        <div className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-2">Time Complexity</div>
                                        <code className="text-[#004040] font-mono">{doc.complexity.time}</code>
                                    </CardContent>
                                </Card>
                                <Card className="bg-white border-gray-200 shadow-sm">
                                    <CardContent className="p-6">
                                        <div className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-2">Space Complexity</div>
                                        <code className="text-[#004040] font-mono">{doc.complexity.space}</code>
                                    </CardContent>
                                </Card>
                            </div>
                        </section>

                        {/* Usage Guidelines */}
                        <section id="guidelines">
                            <h2 className="text-2xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                                <Lightbulb className="w-6 h-6 text-[#004040]" />
                                Usage Guidelines
                            </h2>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                                {/* When to Use */}
                                <Card className="bg-white border-gray-200 shadow-sm">
                                    <CardHeader className="pb-3">
                                        <CardTitle className="flex items-center gap-2 text-base text-emerald-700">
                                            <CheckCircle2 className="w-5 h-5" />
                                            When to Use
                                        </CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <ul className="space-y-2">
                                            {doc.whenToUse.map((item, i) => (
                                                <li key={i} className="flex items-start gap-2 text-sm text-gray-600">
                                                    <span className="text-emerald-500 mt-0.5">✓</span>
                                                    {item}
                                                </li>
                                            ))}
                                        </ul>
                                    </CardContent>
                                </Card>

                                {/* When Not to Use */}
                                <Card className="bg-white border-gray-200 shadow-sm">
                                    <CardHeader className="pb-3">
                                        <CardTitle className="flex items-center gap-2 text-base text-rose-700">
                                            <XCircle className="w-5 h-5" />
                                            When NOT to Use
                                        </CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <ul className="space-y-2">
                                            {doc.whenNotToUse.map((item, i) => (
                                                <li key={i} className="flex items-start gap-2 text-sm text-gray-600">
                                                    <span className="text-rose-500 mt-0.5">✗</span>
                                                    {item}
                                                </li>
                                            ))}
                                        </ul>
                                    </CardContent>
                                </Card>
                            </div>

                            {/* Tips */}
                            <Card className="bg-amber-50 border-amber-200">
                                <CardHeader className="pb-3">
                                    <CardTitle className="flex items-center gap-2 text-base text-amber-800">
                                        <Lightbulb className="w-5 h-5" />
                                        Pro Tips
                                    </CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <ul className="space-y-2">
                                        {doc.tips.map((tip, i) => (
                                            <li key={i} className="flex items-start gap-2 text-sm text-amber-900">
                                                <span className="text-amber-600 mt-0.5">💡</span>
                                                {tip}
                                            </li>
                                        ))}
                                    </ul>
                                </CardContent>
                            </Card>
                        </section>

                        {/* Examples */}
                        <section id="examples">
                            <h2 className="text-2xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                                <Database className="w-6 h-6 text-[#004040]" />
                                Examples
                            </h2>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <Card className="bg-white border-gray-200 shadow-sm">
                                    <CardHeader className="pb-2">
                                        <CardTitle className="text-sm text-gray-700">Sample Input (CSV)</CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <pre className="p-4 bg-gray-100 rounded-lg text-xs font-mono text-gray-700 overflow-x-auto">
                                            {doc.exampleDataset}
                                        </pre>
                                    </CardContent>
                                </Card>
                                <Card className="bg-white border-gray-200 shadow-sm">
                                    <CardHeader className="pb-2">
                                        <CardTitle className="text-sm text-gray-700">Sample Output (JSON)</CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <pre className="p-4 bg-gray-900 rounded-lg text-xs font-mono text-emerald-400 overflow-x-auto">
                                            {doc.exampleOutput}
                                        </pre>
                                    </CardContent>
                                </Card>
                            </div>
                        </section>

                        {/* References */}
                        <section id="references">
                            <h2 className="text-2xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                                <ExternalLink className="w-6 h-6 text-[#004040]" />
                                References & Further Reading
                            </h2>
                            <Card className="bg-white border-gray-200 shadow-sm">
                                <CardContent className="p-6 space-y-3">
                                    {doc.references.map((ref, i) => (
                                        <a
                                            key={i}
                                            href={ref.url}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="flex items-center justify-between p-3 rounded-lg hover:bg-gray-50 border border-gray-100 transition-colors group"
                                        >
                                            <span className="text-sm text-gray-700 group-hover:text-[#004040]">
                                                {ref.title}
                                            </span>
                                            <ExternalLink className="w-4 h-4 text-gray-400 group-hover:text-[#004040]" />
                                        </a>
                                    ))}
                                </CardContent>
                            </Card>
                        </section>

                        {/* Related Algorithms */}
                        <section>
                            <h2 className="text-2xl font-bold text-gray-900 mb-4">Related Algorithms</h2>
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                {algorithms
                                    .filter((a) => a.id !== id && a.category === algorithm.category)
                                    .slice(0, 3)
                                    .map((algo) => (
                                        <Link key={algo.id} href={`/docs/${algo.id}`}>
                                            <Card className="bg-white border-gray-200 shadow-sm hover:shadow-md transition-shadow h-full">
                                                <CardContent className="p-4">
                                                    <h3 className="font-semibold text-gray-900 mb-1">{algo.name}</h3>
                                                    <p className="text-xs text-gray-500 line-clamp-2">{algo.description}</p>
                                                    <div className="flex items-center gap-1 mt-3 text-xs text-[#004040]">
                                                        <span>Read docs</span>
                                                        <ArrowRight className="w-3 h-3" />
                                                    </div>
                                                </CardContent>
                                            </Card>
                                        </Link>
                                    ))}
                            </div>
                        </section>
                    </div>
                </div>
            </div>
        </div>
    );
}
