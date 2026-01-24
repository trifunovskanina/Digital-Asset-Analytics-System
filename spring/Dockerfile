FROM eclipse-temurin:21-jdk AS build
WORKDIR /build
COPY analytics/mvnw analytics/mvnw.cmd analytics/pom.xml ./
COPY analytics/.mvn .mvn
RUN chmod +x mvnw
RUN ./mvnw -B -q dependency:go-offline || true
COPY analytics/src src
RUN ./mvnw -B -DskipTests package

FROM eclipse-temurin:21-jre
WORKDIR /app
COPY --from=build /build/target/analytics-*.jar app.jar
EXPOSE 8080
ENTRYPOINT ["java", "-jar", "app.jar"]
