import { NavLink } from "react-router-dom";
import { BarChart3, FileText, Home, Info, Layers } from "lucide-react";
import { cn } from "@/lib/utils";

const NAV = [
  { to: "/",          label: "Tahmin",         icon: Home },
  { to: "/dashboard", label: "Dashboard",      icon: BarChart3 },
  { to: "/batch",     label: "Toplu",          icon: Layers },
  { to: "/errors",    label: "Hata Analizi",   icon: FileText },
  { to: "/about",     label: "Hakkında",       icon: Info },
];

export default function Navbar() {
  return (
    <header className="sticky top-0 z-30 border-b border-gray-200 bg-white/90 backdrop-blur">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between gap-4 px-4 sm:px-6">
        <NavLink to="/" className="flex items-center gap-2">
          <div className="grid h-8 w-8 place-items-center rounded-md bg-primary text-sm font-bold text-white">
            TR
          </div>
          <span className="hidden text-sm font-semibold text-gray-900 sm:inline">
            TR Otel Sentiment
          </span>
        </NavLink>
        <nav className="flex items-center gap-1 overflow-x-auto">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                cn(
                  "inline-flex items-center gap-1.5 whitespace-nowrap rounded-md px-3 py-1.5 text-sm font-medium",
                  isActive
                    ? "bg-primary-50 text-primary-800"
                    : "text-gray-600 hover:bg-gray-100",
                )
              }
            >
              <Icon className="h-4 w-4" />
              <span className="hidden sm:inline">{label}</span>
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  );
}
