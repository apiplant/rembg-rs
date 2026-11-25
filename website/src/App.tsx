import { Footer } from "./components/Footer";
import { Header } from "./components/Header";
import { Home } from "./components/Home";
import { Docs } from "./components/Docs";

export function App() {
  const isDocs = window.location.pathname.replace(/\/+$/, "") === "/docs";
  return (
    <>
      <Header docs={isDocs} />
      <main id="top">
        {isDocs ? <Docs /> : <Home />}
      </main>
      <Footer />
    </>
  );
}
