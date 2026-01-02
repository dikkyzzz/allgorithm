"use client";

import React, { useState } from "react";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import { algorithms } from "@/lib/algorithms";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Search, ArrowUpRight, Grid3X3, List } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Header } from "@/components/landing/Header";

const categoryColors: Record<string, string> = {
    'Clustering': 'bg-blue-100 text-blue-700',
    'Classification': 'bg-violet-100 text-violet-700',
    'Regression': 'bg-emerald-100 text-emerald-700',
    'Dimensionality Reduction': 'bg-amber-100 text-amber-700',
    'Text Mining': 'bg-rose-100 text-rose-700',
    'Sorting': 'bg-orange-100 text-orange-700',
    'Searching': 'bg-cyan-100 text-cyan-700',
    'Graph': 'bg-indigo-100 text-indigo-700',
    'Deep Learning': 'bg-purple-100 text-purple-700',
    'Statistics': 'bg-teal-100 text-teal-700',
    'Security': 'bg-red-100 text-red-700',
    'Recommendation': 'bg-pink-100 text-pink-700',
};

export default function AlgorithmsPage() {
    const [activeCategory, setActiveCategory] = useState("All");
    const [searchQuery, setSearchQuery] = useState("");
    const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');

    const categories = ["All", ...Array.from(new Set(algorithms.map((a) => a.category)))];

    const filteredAlgorithms = algorithms.filter((a) => {
        const matchesCategory = activeCategory === "All" || a.category === activeCategory;
        const matchesSearch = a.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
            a.description.toLowerCase().includes(searchQuery.toLowerCase());
        return matchesCategory && matchesSearch;
    });

    return (
        <div className="min-h-screen bg-gray-50">
            <Header />

            {/* Hero Section */}
            <div className="bg-gradient-to-b from-[#004040] to-[#006060] text-white pt-32 pb-16">
                <div className="container max-w-6xl mx-auto px-4">
                    <h1 className="text-4xl md:text-5xl font-bold mb-4">
                        Algorithm Library
                    </h1>
                    <p className="text-xl text-white/80 mb-8 max-w-2xl">
                        Browse all {algorithms.length} algorithms across {categories.length - 1} categories.
                        Find the perfect algorithm for your use case.
                    </p>

                    {/* Search Bar */}
                    <div className="relative max-w-xl">
                        <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                        <Input
                            placeholder="Search algorithms..."
                            className="pl-12 py-6 text-lg bg-white text-gray-900 border-0 rounded-full shadow-lg"
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                        />
                    </div>
                </div>
            </div>

            {/* Main Content */}
            <div className="container max-w-6xl mx-auto px-4 py-12">
                {/* Category Filter */}
                <div className="flex flex-wrap items-center justify-between gap-4 mb-8">
                    <div className="flex flex-wrap gap-2">
                        {categories.map((cat) => {
                            const count = cat === 'All'
                                ? algorithms.length
                                : algorithms.filter(a => a.category === cat).length;
                            return (
                                <button
                                    key={cat}
                                    onClick={() => setActiveCategory(cat)}
                                    className={`
                                        px-4 py-2 rounded-full text-sm font-medium transition-all duration-200
                                        ${activeCategory === cat
                                            ? 'bg-[#004040] text-white shadow-lg'
                                            : 'bg-white text-gray-600 hover:bg-gray-100 border border-gray-200'
                                        }
                                    `}
                                >
                                    {cat}
                                    <span className={`ml-2 text-xs ${activeCategory === cat ? 'text-white/70' : 'text-gray-400'}`}>
                                        {count}
                                    </span>
                                </button>
                            );
                        })}
                    </div>

                    {/* View Toggle */}
                    <div className="flex gap-1 bg-white border border-gray-200 rounded-lg p-1">
                        <Button
                            variant={viewMode === 'grid' ? 'default' : 'ghost'}
                            size="sm"
                            onClick={() => setViewMode('grid')}
                            className={viewMode === 'grid' ? 'bg-[#004040]' : ''}
                        >
                            <Grid3X3 className="w-4 h-4" />
                        </Button>
                        <Button
                            variant={viewMode === 'list' ? 'default' : 'ghost'}
                            size="sm"
                            onClick={() => setViewMode('list')}
                            className={viewMode === 'list' ? 'bg-[#004040]' : ''}
                        >
                            <List className="w-4 h-4" />
                        </Button>
                    </div>
                </div>

                {/* Results Count */}
                <div className="text-sm text-gray-500 mb-6">
                    Showing {filteredAlgorithms.length} of {algorithms.length} algorithms
                </div>

                {/* Algorithms Grid/List */}
                {viewMode === 'grid' ? (
                    <motion.div
                        layout
                        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6"
                    >
                        <AnimatePresence mode="popLayout">
                            {filteredAlgorithms.map((algo) => (
                                <motion.div
                                    key={algo.id}
                                    layout
                                    initial={{ opacity: 0, scale: 0.9 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    exit={{ opacity: 0, scale: 0.9 }}
                                    transition={{ duration: 0.2 }}
                                >
                                    <Link href={`/algorithms/${algo.id}`}>
                                        <Card className="h-full bg-white border-gray-200 hover:border-[#004040]/50 hover:shadow-lg transition-all group cursor-pointer">
                                            <CardHeader>
                                                <div className="flex justify-between items-start mb-2">
                                                    <Badge className={categoryColors[algo.category] || 'bg-gray-100 text-gray-600'}>
                                                        {algo.category}
                                                    </Badge>
                                                    <ArrowUpRight className="w-4 h-4 text-gray-400 group-hover:text-[#004040] transition-colors" />
                                                </div>
                                                <CardTitle className="text-xl text-gray-900 group-hover:text-[#004040] transition-colors">
                                                    {algo.name}
                                                </CardTitle>
                                            </CardHeader>
                                            <CardContent>
                                                <CardDescription className="text-gray-600 line-clamp-2">
                                                    {algo.description}
                                                </CardDescription>
                                            </CardContent>
                                            <CardFooter className="pt-0">
                                                <div className="text-xs text-gray-500 italic">
                                                    {algo.bestFor}
                                                </div>
                                            </CardFooter>
                                        </Card>
                                    </Link>
                                </motion.div>
                            ))}
                        </AnimatePresence>
                    </motion.div>
                ) : (
                    <div className="space-y-3">
                        <AnimatePresence mode="popLayout">
                            {filteredAlgorithms.map((algo) => (
                                <motion.div
                                    key={algo.id}
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: -20 }}
                                    transition={{ duration: 0.2 }}
                                >
                                    <Link href={`/algorithms/${algo.id}`}>
                                        <div className="bg-white border border-gray-200 rounded-lg p-4 hover:border-[#004040]/50 hover:shadow-md transition-all group cursor-pointer flex items-center gap-4">
                                            <Badge className={`${categoryColors[algo.category] || 'bg-gray-100 text-gray-600'} shrink-0`}>
                                                {algo.category}
                                            </Badge>
                                            <div className="flex-1 min-w-0">
                                                <h3 className="font-semibold text-gray-900 group-hover:text-[#004040] transition-colors">
                                                    {algo.name}
                                                </h3>
                                                <p className="text-sm text-gray-500 truncate">
                                                    {algo.description}
                                                </p>
                                            </div>
                                            <ArrowUpRight className="w-5 h-5 text-gray-400 group-hover:text-[#004040] transition-colors shrink-0" />
                                        </div>
                                    </Link>
                                </motion.div>
                            ))}
                        </AnimatePresence>
                    </div>
                )}

                {/* Empty State */}
                {filteredAlgorithms.length === 0 && (
                    <div className="text-center py-16">
                        <div className="text-6xl mb-4">🔍</div>
                        <h3 className="text-xl font-semibold text-gray-900 mb-2">No algorithms found</h3>
                        <p className="text-gray-500">
                            Try adjusting your search or filter criteria
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
}
