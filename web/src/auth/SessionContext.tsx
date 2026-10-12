import { createContext, useContext } from "react";
import type { Me } from "../api/types";

export interface Session {
  me: Me;
  signOut: () => void;
}

export const SessionContext = createContext<Session | null>(null);

export function useSession(): Session {
  const session = useContext(SessionContext);
  if (!session) throw new Error("useSession outside a signed-in session");
  return session;
}
