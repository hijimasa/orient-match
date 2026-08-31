# OrientMatch

**日本語** | [English](README.md)

OrientMatch は、密な勾配方向場の相関と姿勢の粗密探索という既存要素を、扱いやすい
C++17/OpenCV ライブラリとしてまとめたものです。再利用可能なマッチャーAPIと、適用条件を
明示した実務指向の実装です。

対応する問題設定は次の範囲です。

- グレースケールの単一テンプレート
- 既知・固定スケール
- 大きいグレースケール画像中の最良一致1件
- 平行移動と面内回転
- CPU実行。OpenMPがあれば角度候補を並列評価

使用する方向場は

```text
z = (gx, gy) / (sqrt(gx^2 + gy^2) + eps)
```

です。2成分の非中心化相関をエネルギーで正規化し、縮小画像で全位置・粗角度を探索した
後、上位角度候補の位置と近傍角度を原寸画像で精密化します。

## 位置づけ

これは初期段階の参照ライブラリであり、**新規アルゴリズムやSOTA性能を主張するものでは
ありません。** 勾配方向類似度、回転テンプレート群、画像ピラミッド上の姿勢探索には、
Stegerのshape-based matching、LINE-2D、Konishiらのoriented-gradient matchingなどの
先行研究があります。

本ライブラリの価値は、既存要素を次の形で実装・パッケージ化している点にあります。

- 1枚のテンプレートから回転テンプレート群を構築する、再利用可能な`Matcher`
- 疎な量子化方向ではなく、密な連続2成分方向場
- 勾配強度のソフトゲートと、パッチ全体のエネルギー正規化相関
- OpenCVを使ったCPU実装と、任意のOpenMP角度並列化
- 座標・角度規約、入力検証、テスト、CMakeインストールの明文化
- 固定スケール・最良姿勢1件に絞った小さなAPI

これらは設計空間上の一つの実用的な構成ですが、個々の要素や手法群そのものを新規技術
として提示するものではありません。

コードは [radon-template-matching](https://github.com/hijimasa/radon-template-matching) の
画像空間対照 `ori_img` から独立ライブラリとして抽出したものです。

## 関連実装と選び分け

以下の実装はOrientMatchと一部が重なりますが、表現や対象用途が異なります。

| 実装 | 共通点 | OrientMatchとの主な違い |
|---|---|---|
| [OpenCV `matchTemplate`](https://docs.opencv.org/4.x/de/da9/tutorial_template_matching.html) | 画像空間の密な相関 | 固定テンプレートの平行移動を探索する。回転と粗密探索の組み立ては利用側が行う。 |
| [OpenCV LINEMOD](https://docs.opencv.org/4.x/d7/d07/classcv_1_1linemod_1_1Detector.html) | 勾配方向とテンプレートピラミッド | 選択・量子化した方向特徴を使い、通常は必要な視点・姿勢ごとのテンプレートを登録する。 |
| [`shape_based_matching`](https://github.com/meiqua/shape_based_matching) | C++/OpenCV、勾配方向、ピラミッド、回転・スケールテンプレート | LINEMOD/HALCON系の疎特徴照合で、姿勢テンプレート生成、SIMD、NMSを含む検出寄りの実装。 |
| [OpenCV `GeneralizedHoughGuil`](https://docs.opencv.org/4.x/d3/d20/classcv_1_1GeneralizedHoughGuil.html) | エッジ勾配と位置・回転・スケール推定 | 密な正規化相関ではなく、一般化Hough投票を使う。 |
| [`batchmatch`](https://github.com/wlruys/batchmatch) | 正規化勾配場と変換探索 | PyTorch/FFTを中心としたアフィン格子の一括レジストレーションで、小さなC++局所精密化マッチャーではない。 |
| [`corrmatch-rs`](https://github.com/VitalyVorobyev/corrmatch-rs) | 回転対応の粗密テンプレート探索 | OpenCVの密な勾配方向相関ではなく、Rustで輝度ZNCC/SSDを使う。 |

この一覧は網羅性を主張するものではありません。本ライブラリの実務上の区別は、1枚の
テンプレートを受け取るC++ API、密な連続方向場、正規化相関、位置・回転の粗密探索を
一つにまとめている点です。新しい用途では、どれかが常に高速・高精度と仮定せず、上記の
候補を実データで比較してください。

## 必要環境

- CMake 3.16以上
- C++17コンパイラ
- OpenCV (`core`, `imgproc`, `imgcodecs`)
- OpenMP（任意）

## ビルドとテスト

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CLIの実行例:

```bash
./build/orient_match_cli search.png template.png
./build/orient_match_cli search.png template.png -45 90
```

## C++ API

```cpp
#include <opencv2/imgcodecs.hpp>
#include <orient_match/orient_match.hpp>

cv::Mat image = cv::imread("search.png", cv::IMREAD_GRAYSCALE);
cv::Mat templ = cv::imread("template.png", cv::IMREAD_GRAYSCALE);

orient_match::MatcherOptions options;
options.coarse_scale = 0.5;
options.coarse_angle_step_deg = 3.0;
options.fine_angle_step_deg = 1.0;
options.refine_top_k = 5;

// 構築時にテンプレート方向場と粗角度の回転テンプレート群を事前計算する。
orient_match::Matcher matcher(templ, options);
orient_match::MatchResult result = matcher.match(image);

if (result) {
    std::cout << result.center.x << ", " << result.center.y << "\n";
    std::cout << result.angle_deg << " deg, score=" << result.score << "\n";
}
```

単一チャンネルのテンプレートマスクも指定できます。

```cpp
orient_match::Matcher matcher(templ, mask, options);
```

マスクが0の画素は除外されます。マスク境界を偽エッジにしないため、勾配計算後にマスクを
適用します。

インストール後はCMakeから利用できます。

```cmake
find_package(OrientMatch CONFIG REQUIRED)
target_link_libraries(my_program PRIVATE OrientMatch::orient_match)
```

## 座標・角度規約

- 左上画素の中心を `(0, 0)` とするOpenCV座標です。
- `MatchResult::center` は入力画像中のテンプレート中心の絶対座標です。
- 正角度は `cv::getRotationMatrix2D` と同じ反時計回りです。
- 返却角度は `[0, 360)` に正規化します。
- 角度探索範囲は半開区間
  `[angle_start_deg, angle_start_deg + angle_extent_deg)` です。

## アルゴリズム

1. 画像とテンプレートを平滑化し、Sobel勾配を計算する。
2. 各勾配ベクトルを画像適応型のソフトゲートで正規化する。
3. テンプレート方向場にガウシアン窓と任意のマスクを適用する。
4. 縮小画像上で、設定した全粗角度と全有効位置の相関を計算する。
5. 粗角度候補の上位 `refine_top_k` 件を保持する。
6. 原寸画像上で局所位置と近傍の細角度を探索する。
7. 正規化方向場相関が最大の姿勢を返す。

`Matcher` の構築時に粗角度の回転テンプレート群を作るため、同じテンプレートを複数フレーム
へ繰り返し適用できます。構築後は不変で、`match()` の作業バッファは呼び出しごとに独立
しています。

## 適用範囲と現在の制約

- スケール探索は行わない。
- 返すのは最良一致1件のみで、複数物体検出やNMSは未実装。
- 位置と角度は離散値で、サブピクセル・サブ角度補間は未実装。
- 粗密探索はヒューリスティックであり、真の大域最大を取りこぼさない保証はない。
- スコアは校正済み確率でも、あらゆる入力に共通する検出閾値でもない。
- テンプレートの正方形回転キャンバス全体が探索画像内に収まる必要がある。
- 精密段では原寸テンプレートを候補角度ごとに回転するため、大テンプレートや広い探索は
  高コストになり得る。
- 現時点では、特徴点ベース照合、学習型検出器、固定スケールを超える一般のアフィン位置
  合わせを置き換えることは目的としていない。

## ライセンス

MIT。詳細は [LICENSE](LICENSE) を参照してください。
