import { Header } from "@/components/landing/Header";
import { Hero } from "@/components/landing/Hero";
import { Stats } from "@/components/landing/Stats";
import { AlgorithmExplorer } from "@/components/landing/AlgorithmExplorer";
import { Demo } from "@/components/landing/Demo";
import { HowItWorks } from "@/components/landing/HowItWorks";
import { UseCases } from "@/components/landing/UseCases";
import { FAQ } from "@/components/landing/FAQ";
import { Footer } from "@/components/landing/Footer";

export default function Home() {
  return (
    <div className="min-h-screen bg-zinc-950 text-white selection:bg-cyan-500/30">
      <Header />
      <main>
        <Hero />
        <Stats />
        <AlgorithmExplorer />
        <Demo />
        <HowItWorks />
        <UseCases />
        <FAQ />
      </main>
      <Footer />
    </div>
  );
}
