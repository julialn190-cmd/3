import Replicate from "replicate";
import sharp from "sharp";

const MODEL =
  "meta/sam-2:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83";

const NAMES = [
  "avant",
  "arriere",
  "gauche",
  "droite",
  "dessus",
  "dessous"
];

function parseDataUrl(dataUrl) {
  const m = /^data:image\/[a-zA-Z0-9.+-]+;base64,(.+)$/.exec(dataUrl || "");
  if (!m) throw new Error("Image reçue invalide.");
  return Buffer.from(m[1], "base64");
}

async function chooseBestMask(urls) {
  let best = null;
  let bestScore = -Infinity;

  for (const url of urls || []) {
    const res = await fetch(String(url));
    if (!res.ok) continue;

    const buf = Buffer.from(await res.arrayBuffer());

    const { data, info } = await sharp(buf)
      .resize(190, 190, { fit: "fill" })
      .grayscale()
      .raw()
      .toBuffer({ resolveWithObject: true });

    for (const invert of [false, true]) {
      let area = 0;
      let sx = 0;
      let sy = 0;
      let border = 0;

      const w = info.width;
      const h = info.height;

      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const raw = data[y * w + x];
          const fg = invert ? raw < 128 : raw >= 128;

          if (!fg) continue;

          area++;
          sx += x;
          sy += y;

          if (x < 4 || y < 4 || x >= w - 4 || y >= h - 4) {
            border++;
          }
        }
      }

      if (area < 200) continue;

      const frac = area / (w * h);

      if (frac < 0.03 || frac > 0.82) continue;

      const cx = sx / area;
      const cy = sy / area;

      const dx = (cx - (w - 1) / 2) / (w / 2);
      const dy = (cy - (h - 1) / 2) / (h / 2);

      const center =
        1 - Math.min(1, Math.hypot(dx, dy));

      const borderFrac = border / area;

      const areaPreference =
        1 - Math.min(1, Math.abs(frac - 0.30) / 0.35);

      const score =
        2.2 * center +
        1.5 * areaPreference -
        4.0 * borderFrac;

      if (score > bestScore) {
        const out = Buffer.alloc(w * h);

        for (let i = 0; i < w * h; i++) {
          const fg = invert ? data[i] < 128 : data[i] >= 128;
          out[i] = fg ? 255 : 0;
        }

        bestScore = score;
        best = {
          out,
          w,
          h
        };
      }
    }
  }

  if (!best) {
    throw new Error("SAM2 n’a pas produit de masque exploitable.");
  }

  return sharp(best.out, {
    raw: {
      width: best.w,
      height: best.h,
      channels: 1
    }
  })
    .png()
    .toBuffer();
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({
      error: "Méthode non autorisée."
    });
  }

  if (!process.env.REPLICATE_API_TOKEN) {
    return res.status(500).json({
      error: "REPLICATE_API_TOKEN n’est pas configuré dans Vercel."
    });
  }

  try {
    const input = parseDataUrl(req.body?.image);

    const meta = await sharp(input).metadata();

    if (!meta.width || !meta.height) {
      throw new Error("Dimensions de l’image introuvables.");
    }

    const cellW = Math.floor(meta.width / 3);
    const cellH = Math.floor(meta.height / 2);

    const replicate = new Replicate({
      auth: process.env.REPLICATE_API_TOKEN
    });

    const masks = [];

    for (let row = 0; row < 2; row++) {
      for (let col = 0; col < 3; col++) {
        const mx = Math.floor(cellW * 0.025);
        const my = Math.floor(cellH * 0.035);

        const left = col * cellW + mx;
        const top = row * cellH + my;

        const width = Math.max(8, cellW - 2 * mx);
        const height = Math.max(8, cellH - 2 * my);

        const crop = await sharp(input)
          .extract({
            left,
            top,
            width,
            height
          })
          .resize(512, 512, {
            fit: "fill"
          })
          .jpeg({
            quality: 82
          })
          .toBuffer();

        const dataUrl =
          "data:image/jpeg;base64," +
          crop.toString("base64");

        const output = await replicate.run(MODEL, {
          input: {
            image: dataUrl,
            use_m2m: true,
            points_per_side: 32,
            pred_iou_thresh: 0.86,
            stability_score_thresh: 0.92
          }
        });

        const urls = (output?.individual_masks || []).map(String);

        if (!urls.length) {
          throw new Error(
            `Aucun masque pour la vue ${NAMES[masks.length]}.`
          );
        }

        const selected = await chooseBestMask(urls);

        masks.push(
          "data:image/png;base64," +
          selected.toString("base64")
        );
      }
    }

    return res.status(200).json({
      ok: true,
      masks
    });

  } catch (err) {
    console.error(err);

    return res.status(500).json({
      error:
        err?.message ||
        "Erreur pendant l’analyse IA."
    });
  }
           }
