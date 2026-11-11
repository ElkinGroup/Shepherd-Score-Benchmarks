/*
 * PheSAScreen — a thin CLI around OpenChemLib's PheSA (Pharmacophore-Enhanced Shape
 * Alignment, Wahl & Sander, JCIM 2024, openchemlib 2024.4.2), written for the fig2 CPU
 * throughput benchmark. PheSA ships only as a Java library (no command-line entry point),
 * so this is the analog of the `shaep` / `align_it` binaries the other CPU adapters call.
 *
 * ONE JVM launch scores a whole benchmark cell (pairwise OR screen): every unique query and
 * db molecule is featurized ONCE (descriptors reused across pairs, like shaep's field grids /
 * align_it's pharmacophores), so JVM startup and descriptor build are not re-paid per pair.
 *
 * Invocation:
 *   java -cp openchemlib-2024.4.2.jar:. PheSAScreen \
 *        --dbase d.sdf  (--queries q.sdf | --reference q.sdf) \
 *        [--pairs pairs.tsv] --scores out.tab [--shape-only] [--threads T]
 *
 *   SCREEN  (no --pairs): score query #0 against every db record.
 *           output line: "<dbName>\t<similarity>"
 *   PAIRS   (--pairs tsv, lines "qIndex<TAB>dIndex" into the q/d SDF record order): score
 *           exactly those (query, db) pairs.   output line: "<qIndex>\t<dIndex>\t<similarity>"
 *
 * FAIR-COMPARISON CHOICES
 *   * DescriptorHandlerShapeOneConf — SINGLE-conformer (rigid) PheSA. Every other tool in the
 *     panel aligns the one embedded conformer it is handed; PheSA's default handler instead
 *     generates a conformer ENSEMBLE per molecule and aligns all of them (its published
 *     screening mode), which would be N-fold more work per "alignment" and not comparable as
 *     a per-pair throughput. OneConf aligns exactly the input pose, one rigid optimization
 *     per pair — the apples-to-apples unit.
 *   * ppWeight selects the channel: 0.0 => pure shape (the fss `vol` comparator); the library
 *     default (0.5) => shape + pharmacophore overlap, PheSA's native score (`vol_color`); 1.0 =>
 *     pure directional pharmacophore (`pharm`). It is NOT merely a scoring reweight: ppWeight
 *     drives the OPTIMIZER OBJECTIVE too (EvaluableOverlap: (1-w)*shapeOverlap + w*ppOverlap, and
 *     the gradient likewise) as well as the reported similarity. The descriptor itself is
 *     ppWeight-INDEPENDENT, and PheSA's search effort (12 PMI seeds + pharmacophore-triangle
 *     seeds; top 10 + top 50 LBFGS-refined) is held at its default for every weight — so the three
 *     modes differ only in the objective, never in how hard PheSA looks. See handler() for the two
 *     openchemlib ctor bugs that made an EARLIER version of this file silently run `vol`/`pharm`
 *     as 200-conformer ENSEMBLES (~800x slower, and not the input pose) while `vol_color` ran
 *     rigid. All three now go through the same rigid, public-API path.
 *   * --threads is passed explicitly (the SLURM job pins it to $SLURM_CPUS_PER_TASK) so PheSA
 *     does not oversubscribe the cgroup. Each worker uses its OWN handler instance.
 *
 * WHAT IS TIMED: the whole process — read + descriptor build + align. Like shaep/align_it, an
 * external binary cannot separate featurize from align, so its per-alignment cost includes the
 * (cheap, single-conformer) descriptor build. That is the honest real-world cost of the tool.
 */
import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.chem.io.SDFileParser;
import com.actelion.research.chem.phesa.DescriptorHandlerShapeOneConf;
import com.actelion.research.chem.phesa.PheSAMolecule;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

public class PheSAScreen {

    /**
     * Rigid single-conformer handler at the requested ppWeight, built ONLY through the public API.
     *
     * TWO openchemlib-2024.4.2 constructor bugs are dodged here; both are avoided by construction,
     * so nothing is patched, forked, or reflected into:
     *
     *  (1) ONE-CONF BUG.  `DescriptorHandlerShapeOneConf(double ppWeight)` is NOT one-conf:
     *      it chains super(ppWeight) -> DescriptorHandlerShape(double) -> this(false, 200, w),
     *      i.e. singleBaseConformation = FALSE.  createDescriptor() then DISCARDS the input 3D pose
     *      and generates a 200-conformer ensemble, and getSimilarity() overlays refConfs x fitConfs.
     *      Only the NO-ARG ctor chains super(true) and is genuinely rigid.  So we always use it.
     *
     *  (2) PPWEIGHT BUG.  Every ppWeight ctor stores the arg in its own field but builds a DEFAULT
     *      PheSASetting (ppWeight 0.5) and never propagates it.  getSimilarity() scores off the
     *      SETTING, so a ctor-supplied ppWeight is silently ignored.  The public getPhesaSetting()
     *      is the supported way in, and it drives BOTH the alignment objective (EvaluableOverlap:
     *      (1-w)*shapeOverlap + w*ppOverlap, gradient likewise) and the reported similarity.
     *
     * ppWeight: 0.0 = pure Gaussian shape | 0.5 = library default blend | 1.0 = pure pharmacophore.
     * PheSA's own pose seeding (useTriangle=true: pharmacophore-triplet seeds, plus 12 PMI
     * orientations; top 50 + top 10 refined by LBFGS) is left at its DEFAULT for every weight --
     * that is PheSA's native search effort, and holding it fixed across weights is what makes the
     * modes comparable to each other.
     */
    static DescriptorHandlerShapeOneConf handler(boolean shapeOnly, Double ppWeight) {
        DescriptorHandlerShapeOneConf h = new DescriptorHandlerShapeOneConf();  // no-arg == rigid
        h.getPhesaSetting().setPpWeight(effPpWeight(shapeOnly, ppWeight));
        h.getPhesaSetting().setUseTriangle(USE_TRIANGLE);
        return h;
    }

    /**
     * PheSA seeds poses two ways: 12 canonical PMI orientations, and pharmacophore-TRIANGLE matches
     * (top 50 + top 10 then LBFGS-refined). useTriangle defaults to TRUE at every ppWeight, so even
     * ppWeight=0 ("pure shape") takes PHARMACOPHORE-DERIVED starting poses. Seeds only pick which
     * local optimum is found — they never enter the objective, so the score stays a pure Gaussian
     * shape Tanimoto — but it does mean shape-only PheSA is not pharmacophore-BLIND the way fss
     * `vol` is. We keep the default ON: it is PheSA's native search and it can only HELP PheSA, so
     * leaving it is the generous (and defensible) choice; hobbling a competitor to flatter fss would
     * be the unfair one. `--no-triangle` exists purely so the sensitivity of that choice can be
     * measured on demand. NOT used for the published numbers.
     */
    static boolean USE_TRIANGLE = true;

    /** Rigid one-conf invariant + silent-zero detection. Called once per SDF after featurization. */
    static void verifyDescriptors(PheSAMolecule[] descs, String which) {
        int maxConfs = 0, failedDesc = 0, noPP = 0;
        for (PheSAMolecule d : descs) {
            if (d == null || d.getVolumes().isEmpty()) { failedDesc++; continue; }  // incl FAILED_OBJECT
            maxConfs = Math.max(maxConfs, d.getVolumes().size());
            if (d.getVolumes().get(0).getPPGaussians().isEmpty()) noPP++;
        }
        // HARD FAIL: a one-conf descriptor holds exactly ONE volume. More than one means the
        // DescriptorHandlerShapeOneConf(double) ensemble bug has crept back and the benchmark would
        // silently measure a different algorithm (input pose discarded, refConfs x fitConfs overlay)
        // rather than error out. Abort so no regenerated dataset can ever be quietly wrong.
        if (maxConfs > 1)
            throw new IllegalStateException("NOT single-conformer: a " + which + " descriptor holds "
                + maxConfs + " conformers. The OneConf ctor bug is back — handler() must use the "
                + "NO-ARG ctor and set ppWeight via getPhesaSetting().");
        if (failedDesc > 0)
            System.err.println("WARNING: " + failedDesc + "/" + descs.length + " " + which
                + " descriptors FAILED (>200 heavy atoms, no 3D coords, or missing explicit H).");
        // Zero-pharmacophore molecules: openchemlib's ppSimilarity correction factor is
        // nPPGaussians/sum(weights) = 0/0 = NaN, and NaN propagates through
        // sim = (1-w)*atomSim + w*ppSim for EVERY w (0.0*NaN is NaN in Java), so `sim > maxSim`
        // is false at every seed and the pair reports 0.0. That is a silent failure, not a real
        // zero overlap, and it bites shape-only exactly as hard as pharmacophore-only.
        if (noPP > 0)
            System.err.println("WARNING: " + noPP + "/" + descs.length + " " + which + " molecules"
                + " have NO pharmacophore points -> NaN correction factor -> forced 0.0 score in"
                + " EVERY mode (including --shape-only). These are NOT genuine zero overlaps.");
    }

    /** Effective ppWeight actually desired for a (shapeOnly, ppWeight) request. */
    static double effPpWeight(boolean shapeOnly, Double ppWeight) {
        if (shapeOnly)        return 0.0;
        if (ppWeight != null) return ppWeight;
        return 0.5;
    }

    static List<StereoMolecule> readSDF(String path, List<String> namesOut) {
        List<StereoMolecule> mols = new ArrayList<>();
        SDFileParser p = new SDFileParser(path);
        int i = 0;
        while (p.next()) {
            mols.add(p.getMolecule());
            if (namesOut != null) {
                String nm = p.getMoleculeName();
                namesOut.add((nm == null || nm.isEmpty()) ? ("mol" + i) : nm);
            }
            i++;
        }
        return mols;
    }

    /** Build descriptors for every molecule, in parallel (each worker its own handler). */
    static PheSAMolecule[] buildDescriptors(List<StereoMolecule> mols, boolean shapeOnly,
                                            Double ppWeight, ExecutorService ex, int threads)
            throws Exception {
        final int n = mols.size();
        final PheSAMolecule[] out = new PheSAMolecule[n];
        List<Future<?>> futs = new ArrayList<>();
        int chunk = (n + threads - 1) / threads;
        for (int t = 0; t < threads; t++) {
            final int lo = t * chunk, hi = Math.min(n, (t + 1) * chunk);
            if (lo >= hi) break;
            futs.add(ex.submit(() -> {
                DescriptorHandlerShapeOneConf h = handler(shapeOnly, ppWeight);
                for (int i = lo; i < hi; i++) {
                    try { out[i] = h.createDescriptor(mols.get(i)); }
                    catch (Exception e) { out[i] = null; }
                }
            }));
        }
        for (Future<?> f : futs) f.get();
        return out;
    }

    public static void main(String[] args) throws Exception {
        String qPath = null, dPath = null, pairsPath = null, outPath = null;
        boolean shapeOnly = false;
        Double ppWeight = null;
        int threads = Runtime.getRuntime().availableProcessors();
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--queries":
                case "--reference":  qPath = args[++i]; break;
                case "--dbase":      dPath = args[++i]; break;
                case "--pairs":      pairsPath = args[++i]; break;
                case "--scores":     outPath = args[++i]; break;
                case "--shape-only":  shapeOnly = true; break;
                case "--ppweight":    ppWeight = Double.parseDouble(args[++i]); break;
                case "--threads":     threads = Integer.parseInt(args[++i]); break;
                case "--no-triangle": USE_TRIANGLE = false; break;   // sensitivity check only
                default: System.err.println("unknown arg: " + args[i]);
            }
        }
        if (qPath == null || dPath == null || outPath == null)
            throw new IllegalArgumentException("need --queries/--reference, --dbase, --scores");
        threads = Math.max(1, threads);
        System.err.println("PheSA rigid-1conf ppWeight=" + effPpWeight(shapeOnly, ppWeight)
                + " simMode=TANIMOTO directional=true useTriangle=" + USE_TRIANGLE
                + " shapeOnly=" + shapeOnly + " threads=" + threads);

        List<String> dNames = new ArrayList<>();
        List<StereoMolecule> qMols = readSDF(qPath, null);
        List<StereoMolecule> dMols = readSDF(dPath, dNames);

        ExecutorService ex = Executors.newFixedThreadPool(threads);
        final PheSAMolecule[] qDesc = buildDescriptors(qMols, shapeOnly, ppWeight, ex, threads);
        final PheSAMolecule[] dDesc = buildDescriptors(dMols, shapeOnly, ppWeight, ex, threads);
        verifyDescriptors(qDesc, "query");   // aborts if the ensemble ctor bug ever returns
        verifyDescriptors(dDesc, "dbase");

        // Pair list: explicit (--pairs) or screen default (query 0 vs every db record).
        final int[] qi, di;
        if (pairsPath != null) {
            List<int[]> pl = new ArrayList<>();
            try (BufferedReader br = new BufferedReader(new FileReader(pairsPath))) {
                String ln;
                while ((ln = br.readLine()) != null) {
                    ln = ln.trim();
                    if (ln.isEmpty()) continue;
                    String[] p = ln.split("[\\t ,]+");
                    pl.add(new int[]{Integer.parseInt(p[0]), Integer.parseInt(p[1])});
                }
            }
            qi = new int[pl.size()]; di = new int[pl.size()];
            for (int k = 0; k < pl.size(); k++) { qi[k] = pl.get(k)[0]; di[k] = pl.get(k)[1]; }
        } else {
            qi = new int[dMols.size()]; di = new int[dMols.size()];
            for (int j = 0; j < dMols.size(); j++) { qi[j] = 0; di[j] = j; }
        }

        final int m = qi.length;
        final float[] sims = new float[m];
        final boolean so = shapeOnly;
        final Double pw = ppWeight;
        final AtomicInteger failed = new AtomicInteger();
        List<Future<?>> futs = new ArrayList<>();
        int chunk = (m + threads - 1) / threads;
        for (int t = 0; t < threads; t++) {
            final int lo = t * chunk, hi = Math.min(m, (t + 1) * chunk);
            if (lo >= hi) break;
            futs.add(ex.submit(() -> {
                DescriptorHandlerShapeOneConf h = handler(so, pw);   // NOTE: never getThreadSafeCopy()
                for (int k = lo; k < hi; k++) {                      // -- it resets the PheSASetting.
                    PheSAMolecule a = qDesc[qi[k]], b = dDesc[di[k]];
                    float s = 0f;
                    // FAILED_OBJECT is a real, NON-NULL PheSAMolecule with zero volumes (thrown by
                    // >200 heavy atoms, missing 3D coords, or -- in one-conf mode -- missing explicit
                    // hydrogens). A null check alone sails past it and scores a failure as a 0, which
                    // is indistinguishable from a genuine no-overlap. Count them and FLAG instead.
                    if (a == null || b == null || h.calculationFailed(a) || h.calculationFailed(b)) {
                        failed.incrementAndGet();
                    } else {
                        try { s = h.getSimilarity(a, b); } catch (Exception e) { s = 0f; }
                    }
                    sims[k] = s;
                }
            }));
        }
        for (Future<?> f : futs) f.get();
        ex.shutdown();
        if (failed.get() > 0)
            System.err.println("WARNING: " + failed.get() + "/" + m
                    + " pairs had a FAILED descriptor and were scored 0.0");

        boolean screen = (pairsPath == null);
        try (BufferedWriter w = new BufferedWriter(new FileWriter(outPath))) {
            for (int k = 0; k < m; k++) {
                if (screen) w.write(dNames.get(di[k]) + "\t" + sims[k] + "\n");
                else        w.write(qi[k] + "\t" + di[k] + "\t" + sims[k] + "\n");
            }
        }
        System.err.println("scored " + m + " pairs (" + qMols.size() + " queries x "
                + dMols.size() + " db, descriptors built once)");
    }
}
