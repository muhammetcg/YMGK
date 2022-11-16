import 'package:flutter/material.dart';
import 'package:mobys_deneme/NavBar.dart';
import 'package:carousel_slider/carousel_slider.dart';
import 'package:smooth_page_indicator/smooth_page_indicator.dart';
import 'package:mobys_deneme/Views/login.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Flutter Demo',
      theme: ThemeData(
        primarySwatch: Colors.red,
      ),
      home: const MyHomePage(title: 'Flutter Demo Home Page'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({Key? key, required this.title}) : super(key: key);

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  final controller = CarouselController();
  int activeIndex = 0;
  final urlImages = [
    'https://www.reklamevreni.net/tema/firmarehberi/uploads/firmalar/kapak/1.png',
    'https://dogaldekor.com/blog/wp-content/uploads/2019/03/visne-mermer-tas-kaplama-mood-board.jpg',
    'http://www.ortadogumermer.com.tr/wp-content/uploads/2019/12/Elaz%C4%B1%C4%9F-Vi%C5%9Fne-736x414.jpg',
    'https://dogaldekor.com/blog/wp-content/uploads/2019/04/dekorasyon-renk-uyumu.jpg',
    'http://www.arelstone.com/tr/images/Urunler/teknik_detay/Rosso_Levanto/buyuk/5.png',
    'https://www.biancocarraramermer.com/asset/images/shop/elazig-visne-mermer-fiyatlari,-cesitleri-m2-fiyati-plaka-gorselleri-damarli-5-.jpg',
    'https://www.elitrestorasyon.com/Content/Arayuz/Files/urun/800x600/0423a85c-c60e-4a4e-aa30-753e83acc0d9.jpg',
    'https://cdn-ceamn.nitrocdn.com/WnUKhXGMIaEoqvOSzZyyyXsXYanAByXs/assets/static/optimized/wp-content/uploads/2020/02/1c6f796b9315e6b0c60133d297fb0ed6.oztas-mermer-granit-ankara-urunlerimiz-granit-ege-bordo-mermer.jpg',
    'https://im.haberturk.com/2016/09/03/1291638_9db0e00e39a15b7899607b8d00bcd2f2_640x640.jpg',
  ];

  @override
  Widget build(BuildContext context) => Scaffold(
        drawer: NavBar(), //sidebar simgesi geldi
        appBar: AppBar(
          centerTitle: true,
          title: Text('MOBYS'),
          flexibleSpace: Container(
            decoration: BoxDecoration(
                gradient: LinearGradient(
                    colors: [Color(0xFF933b39), Color(0xFF50577A)],
                    begin: Alignment.bottomLeft,
                    end: Alignment.topRight)),
          ),
        ),
        body: Center(
            child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            buildImageSlider(),
            const SizedBox(height: 32),
            buildIndicator(),
            const SizedBox(height: 32),
          ],
        )),
      );

  Widget buildImageSlider() => CarouselSlider.builder(
        carouselController: controller,
        options: CarouselOptions(
          height: 400,
          initialPage: 0,
          enlargeCenterPage: true,
          enlargeStrategy: CenterPageEnlargeStrategy.height,
          autoPlayInterval: Duration(seconds: 2),
          onPageChanged: (index, reason) => setState(() => activeIndex = index),
        ),
        itemCount: urlImages.length,
        itemBuilder: (context, index, realIndex) {
          final urlImage = urlImages[index];

          return buildImage(urlImage, index);
        },
      );

  Widget buildImage(String urlImage, int index) => Container(
        color: Colors.grey,
        margin: EdgeInsets.symmetric(horizontal: 40),
        child: Image.network(
          urlImage,
          fit: BoxFit.cover,
        ),
      );

  Widget buildIndicator() => AnimatedSmoothIndicator(
        activeIndex: activeIndex,
        count: urlImages.length,
        onDotClicked: animateToSlide,
        effect: SlideEffect(
          dotWidth: 15,
          dotHeight: 15,
          activeDotColor: Color(0xFF933b39),
          dotColor: Colors.black12,
        ),
      );

  void animateToSlide(int index) => controller.animateToPage(index);

  void next() => controller.nextPage(duration: Duration(microseconds: 500));

  void previous() =>
      controller.previousPage(duration: Duration(microseconds: 500));
}
