import type { ArchitectureGraph } from '../types/architecture_graph';
import type { PaperData } from '../types/paper';
import { irToArchitectureGraph } from './api';

/** The single gate for what the 2D tab may draw.
 *  `diagramAvailable === false` means the backend found no verified graph:
 *  never rebuild one from the IR, because that would show an unverified diagram.
 *  Older payloads without the flag keep their previous behaviour.
 */
export function selectDiagramGraph(paper: PaperData): ArchitectureGraph | undefined {
  if (paper.diagramAvailable === false) return undefined;
  if (paper.architectureGraph && (paper.architectureGraph.nodes?.length ?? 0) > 0) {
    return paper.architectureGraph;
  }
  if (paper.architectureIR && (paper.architectureIR.nodes?.length ?? 0) > 0) {
    return irToArchitectureGraph(paper.architectureIR);
  }
  return undefined;
}
