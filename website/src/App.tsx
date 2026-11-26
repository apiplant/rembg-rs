import { Footer } from "./components/Footer";
import { Header } from "./components/Header";
import { Home } from "./components/Home";

export function App() {
  return (
    <>
      <Header />
      <main id="top">
        <Home />
      </main>
      <Footer />
    </>
  );
}
